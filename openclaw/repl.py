"""Interactive REPL and Python API for the OpenClaw agent."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any

from openclaw.agent.loop import AgentContext, run, stream_run
from openclaw.agent.types import ThinkingLevel, ToolDefinition, ToolResult
from openclaw.config import AppConfig, load_config
from openclaw.context.guard import ContextGuard
from openclaw.memory.embeddings import EmbeddingProvider
from openclaw.memory.search import MemorySearcher, SearchResult
from openclaw.memory.store import MemoryStore
from openclaw.model.provider import ModelProvider
from openclaw.prompt.bootstrap import load_bootstrap_files
from openclaw.session.manager import SessionManager
from openclaw.skills.loader import build_skills_prompt, load_skills
from openclaw.tools.registry import ToolRegistry


def _register_builtins(registry: ToolRegistry, workspace: str) -> None:
    """Register all built-in tools."""
    from openclaw.tools.builtins import (
        apply_patch,
        bash,
        edit,
        image_tool,
        memory_tool,
        pdf_tool,
        process_tool,
        read,
        web_fetch,
        write,
    )

    modules = [
        (read, "fs"),
        (write, "fs"),
        (edit, "fs"),
        (apply_patch, "fs"),
        (bash, "runtime"),
        (process_tool, "runtime"),
        (web_fetch, "web"),
        (pdf_tool, "analysis"),
        (image_tool, "analysis"),
    ]

    for mod, group in modules:
        async def make_executor(m: Any = mod) -> Any:
            async def executor(args: dict[str, Any]) -> ToolResult:
                return await m.execute(args, workspace=workspace)
            return executor

        registry.register(
            mod.DEFINITION,
            asyncio.coroutine(lambda args, m=mod: m.execute(args, workspace=workspace)).__wrapped__
            if False else
            _make_tool_executor(mod, workspace),
            group=group,
        )

    # Memory tools (stubs — wired up with real searcher in Agent class)
    registry.register(memory_tool.SEARCH_DEFINITION, memory_tool.execute_search, group="memory")
    registry.register(memory_tool.SAVE_DEFINITION, memory_tool.execute_save, group="memory")


def _make_tool_executor(mod: Any, workspace: str) -> Any:
    """Create an async executor closure for a tool module."""
    async def executor(args: dict[str, Any]) -> ToolResult:
        return await mod.execute(args, workspace=workspace)
    return executor


class Agent:
    """High-level Python API for the OpenClaw agent."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.provider = ModelProvider(config)
        self.workspace = config.workspace.resolved_dir
        self.workspace.mkdir(parents=True, exist_ok=True)

        # Tool registry
        self.tool_registry = ToolRegistry()
        _register_builtins(self.tool_registry, str(self.workspace))

        # Memory
        self._memory_store: MemoryStore | None = None
        self._memory_searcher: MemorySearcher | None = None
        self._embedding_provider: EmbeddingProvider | None = None

        # Session
        self._sessions: dict[str, SessionManager] = {}

    @classmethod
    def from_config(cls, path: str | Path | None = None) -> Agent:
        """Create an Agent from a config file."""
        config = load_config(path)
        return cls(config)

    def _get_session(self, session_id: str = "default") -> SessionManager:
        if session_id not in self._sessions:
            session_dir = self.config.session.resolved_dir
            self._sessions[session_id] = SessionManager(session_dir, session_id)
        return self._sessions[session_id]

    def _get_memory_searcher(self) -> MemorySearcher | None:
        if self._memory_store is None:
            try:
                db_path = self.config.memory.resolved_dir / "memory.sqlite"
                self._memory_store = MemoryStore(db_path)
                self._embedding_provider = EmbeddingProvider(self.config)
                self._memory_searcher = MemorySearcher(
                    self._memory_store,
                    self._embedding_provider,
                    self.config.memory,
                )
            except Exception:
                return None
        return self._memory_searcher

    def _build_context(
        self,
        session_id: str = "default",
        thinking: ThinkingLevel | None = None,
    ) -> AgentContext:
        session = self._get_session(session_id)

        # Load bootstrap files
        bootstrap_ctx = load_bootstrap_files(
            self.workspace, self.config.bootstrap
        )

        # Load skills
        skill_dirs = [d for d in self.config.skills.resolved_dirs if d.is_dir()]
        skills_snapshot = load_skills(skill_dirs, self.config.skills.max_skills_in_prompt)
        skills_prompt = build_skills_prompt(skills_snapshot, self.config.skills.max_prompt_chars)

        # Wire up memory tools
        searcher = self._get_memory_searcher()
        if searcher:
            self._wire_memory_tools(searcher)

        return AgentContext(
            config=self.config,
            provider=self.provider,
            session=session,
            tool_registry=self.tool_registry,
            context_guard=ContextGuard(self.config.context),
            thinking=thinking or ThinkingLevel.from_str(
                self.config.models.options.get(
                    self.config.models.default,
                    type("", (), {"thinking": "off"})()
                ).thinking
            ),
            workspace_dir=str(self.workspace),
            bootstrap_ctx=bootstrap_ctx,
            skills_prompt=skills_prompt,
            model=self.config.models.default,
        )

    def _wire_memory_tools(self, searcher: MemorySearcher) -> None:
        """Replace memory tool stubs with real implementations."""
        async def memory_search(args: dict[str, Any]) -> ToolResult:
            query = args.get("query", "")
            max_results = int(args.get("max_results", 10))
            if not query:
                return ToolResult(tool_use_id="", content="Error: query is required", is_error=True)
            try:
                results = await searcher.search(query, max_results=max_results)
                if not results:
                    return ToolResult(tool_use_id="", content="No results found")
                lines = []
                for r in results:
                    lines.append(
                        f"[{r.final_score:.3f}] {r.chunk.file_path}:{r.chunk.line_start}-{r.chunk.line_end}\n"
                        f"{r.chunk.text[:700]}"
                    )
                return ToolResult(tool_use_id="", content="\n\n---\n\n".join(lines))
            except Exception as e:
                return ToolResult(tool_use_id="", content=f"Search error: {e}", is_error=True)

        async def memory_save(args: dict[str, Any]) -> ToolResult:
            import datetime
            content = args.get("content", "")
            file_name = args.get("file", f"{datetime.date.today().isoformat()}.md")
            if not content:
                return ToolResult(tool_use_id="", content="Error: content is required", is_error=True)
            try:
                memory_dir = self.config.memory.resolved_dir
                memory_dir.mkdir(parents=True, exist_ok=True)
                file_path = memory_dir / file_name
                with open(file_path, "a", encoding="utf-8") as f:
                    f.write(f"\n{content}\n")
                # Re-index the file
                await searcher.index_file(file_path)
                return ToolResult(tool_use_id="", content=f"Saved to {file_path}")
            except Exception as e:
                return ToolResult(tool_use_id="", content=f"Save error: {e}", is_error=True)

        from openclaw.tools.builtins.memory_tool import SAVE_DEFINITION, SEARCH_DEFINITION
        self.tool_registry.register(SEARCH_DEFINITION, memory_search, group="memory")
        self.tool_registry.register(SAVE_DEFINITION, memory_save, group="memory")

    async def run(
        self,
        message: str,
        *,
        session_id: str = "default",
        thinking: ThinkingLevel | None = None,
    ) -> RunResult:
        """Run the agent with a user message."""
        ctx = self._build_context(session_id, thinking)
        return await run(ctx, message)

    async def stream(
        self,
        message: str,
        *,
        session_id: str = "default",
        thinking: ThinkingLevel | None = None,
    ):
        """Stream the agent response."""
        ctx = self._build_context(session_id, thinking)
        async for chunk in stream_run(ctx, message):
            yield chunk

    def tool(
        self,
        name: str,
        *,
        description: str = "",
        parameters: list[dict[str, Any]] | None = None,
    ):
        """Decorator to register a custom tool.

        Usage:
            @agent.tool("db_query", description="Query the database")
            async def db_query(sql: str) -> str:
                return run_query(sql)
        """
        from openclaw.agent.types import ToolParameter

        def decorator(func):
            params = []
            if parameters:
                for p in parameters:
                    params.append(ToolParameter(**p))

            defn = ToolDefinition(
                name=name,
                description=description or func.__doc__ or "",
                parameters=params,
            )

            async def executor(args: dict[str, Any]) -> ToolResult:
                try:
                    result = await func(**args) if asyncio.iscoroutinefunction(func) else func(**args)
                    return ToolResult(tool_use_id="", content=str(result))
                except Exception as e:
                    return ToolResult(tool_use_id="", content=f"Error: {e}", is_error=True)

            self.tool_registry.register(defn, executor, group="custom")
            return func

        return decorator


# ---------------------------------------------------------------------------
# Interactive REPL
# ---------------------------------------------------------------------------


def main() -> None:
    """Run the interactive REPL."""
    try:
        from rich.console import Console
        from rich.markdown import Markdown
    except ImportError:
        print("Error: rich is required. Install with: pip install rich")
        sys.exit(1)

    console = Console()
    agent = Agent.from_config()

    console.print("[bold]OpenClaw-Py Agent[/bold]")
    console.print(f"Model: {agent.config.models.default}")
    console.print(f"Workspace: {agent.workspace}")
    console.print("Type /quit to exit, /new for new session\n")

    session_id = "repl"

    async def repl_loop() -> None:
        nonlocal session_id

        while True:
            try:
                user_input = console.input("[bold green]> [/bold green]")
            except (EOFError, KeyboardInterrupt):
                console.print("\nGoodbye!")
                break

            if not user_input.strip():
                continue

            if user_input.strip() == "/quit":
                break

            if user_input.strip() == "/new":
                import uuid
                session_id = uuid.uuid4().hex[:8]
                console.print(f"[dim]New session: {session_id}[/dim]")
                continue

            if user_input.strip() == "/compact":
                console.print("[dim]Compacting session...[/dim]")
                # Trigger manual compaction
                ctx = agent._build_context(session_id)
                ctx.session.load()
                from openclaw.session.compaction import compact_session
                entry = await compact_session(
                    ctx.session, ctx.provider, agent.config.compaction,
                    agent.config.context.max_tokens,
                )
                if entry:
                    console.print(f"[dim]Compacted: {entry.tokens_before} → {entry.tokens_after} tokens[/dim]")
                else:
                    console.print("[dim]Nothing to compact[/dim]")
                continue

            # Stream response
            ctx = agent._build_context(session_id)

            def on_stream(text: str) -> None:
                console.print(text, end="", highlight=False)

            ctx.on_stream = on_stream

            def on_tool_start(name: str, args: dict) -> None:
                console.print(f"\n[dim]⚙ {name}({json.dumps(args)[:100]})[/dim]", highlight=False)

            def on_tool_end(name: str, result: ToolResult) -> None:
                status = "✗" if result.is_error else "✓"
                preview = result.content[:100].replace("\n", " ")
                console.print(f"[dim]  {status} {preview}[/dim]", highlight=False)

            import json
            ctx.on_tool_start = on_tool_start
            ctx.on_tool_end = on_tool_end

            result = await run(ctx, user_input)

            if not ctx.on_stream:
                # Fallback: print full result
                if result.text:
                    console.print()
                    console.print(Markdown(result.text))

            console.print()  # newline after response

            if result.error:
                console.print(f"[red]Error: {result.error}[/red]")

            if result.compacted:
                console.print("[dim]Session was compacted[/dim]")

    asyncio.run(repl_loop())


if __name__ == "__main__":
    main()
