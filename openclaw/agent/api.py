"""High-level Python API for the OpenClaw agent."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from openclaw.agent.loop import AgentContext, run, stream_run
from openclaw.agent.types import (
    RunResult,
    ThinkingLevel,
    ToolDefinition,
    ToolParameter,
    ToolResult,
)
from openclaw.config import AppConfig, load_config
from openclaw.context.guard import ContextGuard
from openclaw.cron import CronScheduler, heartbeat_from_file, heartbeat_memory_check, heartbeat_model_ping
from openclaw.hooks import HookRunner
from openclaw.memory.embeddings import EmbeddingProvider
from openclaw.memory.search import MemorySearcher
from openclaw.memory.store import MemoryStore
from openclaw.model.provider import ModelProvider
from openclaw.prompt.bootstrap import load_bootstrap_files
from openclaw.session.lanes import LaneManager
from openclaw.session.manager import SessionManager
from openclaw.skills.loader import build_skills_prompt, load_skills
from openclaw.tools.registry import ToolRegistry


def _make_tool_executor(mod: Any, workspace: str) -> Any:
    """Create an async executor closure for a tool module."""
    async def executor(args: dict[str, Any]) -> ToolResult:
        return await mod.execute(args, workspace=workspace)
    return executor


def _register_builtins(registry: ToolRegistry, workspace: str) -> None:
    """Register all built-in tools."""
    from openclaw.tools.builtins import (
        apply_patch,
        bash,
        edit,
        hancom_tool,
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
        (hancom_tool, "analysis"),
        # image_tool is registered separately in Agent.__init__ (needs provider)
    ]

    for mod, group in modules:
        registry.register(
            mod.DEFINITION,
            _make_tool_executor(mod, workspace),
            group=group,
        )

    # Image tool (stub — wired up with real provider in Agent class)
    registry.register(image_tool.DEFINITION, _make_tool_executor(image_tool, workspace), group="analysis")

    # Memory tools (stubs — wired up with real searcher in Agent class)
    registry.register(memory_tool.SEARCH_DEFINITION, memory_tool.execute_search, group="memory")
    registry.register(memory_tool.SAVE_DEFINITION, memory_tool.execute_save, group="memory")
    registry.register(memory_tool.MEMORY_GET_DEFINITION, memory_tool.execute_memory_get, group="memory")


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

        # Wire image tool with provider for vision model access
        self._wire_image_tool()

        # Register subagent tools
        self._register_subagent_tools()

        # Memory
        self._memory_store: MemoryStore | None = None
        self._memory_searcher: MemorySearcher | None = None
        self._embedding_provider: EmbeddingProvider | None = None

        # Session
        self._sessions: dict[str, SessionManager] = {}

        # Session lanes (per session)
        self._lane_managers: dict[str, LaneManager] = {}

        # Cron/heartbeat scheduler
        self._scheduler = CronScheduler()

    @classmethod
    def from_config(cls, path: str | Path | None = None) -> Agent:
        """Create an Agent from a config file."""
        config = load_config(path)
        return cls(config)

    def get_lane_manager(self, session_id: str = "default") -> LaneManager:
        """Get (or create) the lane manager for a session."""
        if session_id not in self._lane_managers:
            self._lane_managers[session_id] = LaneManager()
        return self._lane_managers[session_id]

    @property
    def scheduler(self) -> CronScheduler:
        """Access the cron/heartbeat scheduler."""
        return self._scheduler

    async def start_heartbeat(self, interval: float = 120.0) -> None:
        """Start default heartbeat tasks (model ping, memory check)."""
        provider = self.provider
        model = self.config.models.default
        memory_dir = str(self.config.memory.resolved_dir)

        async def _ping() -> None:
            await heartbeat_model_ping(provider, model)

        async def _mem_check() -> None:
            await heartbeat_memory_check(memory_dir)

        self._scheduler.register("model_ping", _ping, interval=interval)
        self._scheduler.register("memory_check", _mem_check, interval=interval * 2)

        # HEARTBEAT.md custom schedule (if present)
        heartbeat_md = self.workspace / "HEARTBEAT.md"
        if heartbeat_md.exists():
            agent_ref = self

            async def _heartbeat_file() -> None:
                await heartbeat_from_file(
                    str(heartbeat_md), provider, model, str(self.workspace),
                    agent=agent_ref,
                )
            self._scheduler.register(
                "heartbeat_file", _heartbeat_file, interval=interval * 5
            )

        await self._scheduler.start()

    async def stop_heartbeat(self) -> None:
        """Stop all scheduled tasks."""
        await self._scheduler.stop()

    def _get_session(self, session_id: str = "default") -> SessionManager:
        if session_id not in self._sessions:
            session_dir = self.config.session.resolved_dir
            self._sessions[session_id] = SessionManager(session_dir, session_id)
        return self._sessions[session_id]

    def _get_memory_searcher(self) -> MemorySearcher | None:
        if self._memory_store is None:
            try:
                from openclaw.memory.search import FileWatcher, Reranker

                db_path = self.config.memory.resolved_dir / "memory.sqlite"
                self._memory_store = MemoryStore(db_path)
                self._embedding_provider = EmbeddingProvider(self.config)

                # Fingerprint check: auto-reset index on embedding config change
                fingerprint = MemoryStore.compute_fingerprint(
                    embedding_model=self.config.models.embedding,
                    base_url=self.config.endpoints.embedding.base_url,
                    chunk_size=self.config.memory.chunk_size,
                    chunk_overlap=self.config.memory.chunk_overlap,
                )
                if not self._memory_store.check_fingerprint(fingerprint):
                    import logging
                    logging.getLogger("openclaw.memory").warning(
                        "Embedding config changed, resetting index for full re-indexing"
                    )
                    self._memory_store.reset_index(fingerprint)

                # Create FileWatcher and register existing memory files
                watcher = FileWatcher(debounce_seconds=30.0)
                memory_dir = self.config.memory.resolved_dir
                if memory_dir.is_dir():
                    for md_file in memory_dir.glob("*.md"):
                        watcher.register(md_file)
                # Also watch workspace MEMORY.md
                workspace_memory = self.workspace / "MEMORY.md"
                if workspace_memory.exists():
                    watcher.register(workspace_memory)

                # Reranker (uses same embedding provider as cross-encoder proxy)
                reranker = Reranker(self._embedding_provider)

                self._memory_searcher = MemorySearcher(
                    self._memory_store,
                    self._embedding_provider,
                    self.config.memory,
                    file_watcher=watcher,
                    reranker=reranker,
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

        # Wire up memory tools (once)
        searcher = self._get_memory_searcher()
        if searcher and not getattr(self, "_memory_tools_wired", False):
            self._wire_memory_tools(searcher)
            self._memory_tools_wired = True

        # Build failover manager with configured fallback models
        from openclaw.model.failover import FailoverManager
        failover = FailoverManager(fallback_models=list(self.config.models.fallback))

        return AgentContext(
            config=self.config,
            provider=self.provider,
            session=session,
            tool_registry=self.tool_registry,
            context_guard=ContextGuard(self.config.context),
            failover=failover,
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
            hook_runner=HookRunner(self.config.hooks),
            memory_searcher=searcher,
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
                # Register with FileWatcher for future change detection
                if searcher.file_watcher:
                    searcher.file_watcher.register(file_path)
                return ToolResult(tool_use_id="", content=f"Saved to {file_path}")
            except Exception as e:
                return ToolResult(tool_use_id="", content=f"Save error: {e}", is_error=True)

        from openclaw.tools.builtins.memory_tool import SAVE_DEFINITION, SEARCH_DEFINITION
        self.tool_registry.register(SEARCH_DEFINITION, memory_search, group="memory")
        self.tool_registry.register(SAVE_DEFINITION, memory_save, group="memory")

    def _wire_image_tool(self) -> None:
        """Replace image tool stub with vision-model-enabled implementation."""
        from openclaw.tools.builtins import image_tool

        provider = self.provider
        model = self.config.models.default
        workspace = str(self.workspace)

        async def image_executor(args: dict[str, Any]) -> ToolResult:
            return await image_tool.execute(
                args, workspace=workspace, provider=provider, model=model
            )

        self.tool_registry.register(image_tool.DEFINITION, image_executor, group="analysis")

    def _register_subagent_tools(self) -> None:
        """Register subagent spawn/list/read tools."""
        from openclaw.subagent.spawn import SubagentConfig, SubagentRegistry

        self._subagent_registry = SubagentRegistry(SubagentConfig())

        # Tool: spawn a subagent
        spawn_def = ToolDefinition(
            name="subagent",
            description=(
                "Spawn a subagent to handle a subtask autonomously. "
                "The subagent runs with its own session and returns a result. "
                "Use this to parallelize work or delegate complex subtasks."
            ),
            parameters=[
                ToolParameter(name="task", description="The task for the subagent to perform"),
                ToolParameter(
                    name="model",
                    description="Model to use (default: same as parent)",
                    required=False,
                ),
            ],
        )

        agent_ref = self

        async def spawn_executor(args: dict[str, Any]) -> ToolResult:
            task = args.get("task", "")
            model = args.get("model", "")
            if not task:
                return ToolResult(tool_use_id="", content="Error: task is required", is_error=True)

            # Check spawn limits
            can, reason = agent_ref._subagent_registry.can_spawn(0, "main")
            if not can:
                return ToolResult(tool_use_id="", content=f"Cannot spawn: {reason}", is_error=True)

            # Spawn and run
            entry = agent_ref._subagent_registry.spawn(
                parent_session_key="main",
                task=task,
                depth=0,
                model=model,
            )
            agent_ref._subagent_registry.mark_running(entry.id)

            try:
                result = await agent_ref.run(
                    task,
                    session_id=entry.session_key,
                )
                agent_ref._subagent_registry.mark_completed(
                    entry.id, text=result.text, error=result.error
                )
                output = result.text or ""
                if result.error:
                    output += f"\n\nSubagent error: {result.error}"
                return ToolResult(
                    tool_use_id="",
                    content=f"[Subagent {entry.id} completed]\n\n{output}",
                )
            except Exception as e:
                agent_ref._subagent_registry.mark_completed(entry.id, error=str(e))
                return ToolResult(
                    tool_use_id="",
                    content=f"Subagent failed: {e}",
                    is_error=True,
                )

        self.tool_registry.register(spawn_def, spawn_executor, group="subagent")

        # Batch spawn: run multiple subagents in parallel
        batch_spawn_def = ToolDefinition(
            name="subagent_batch",
            description=(
                "Spawn multiple subagents in parallel. Each task runs concurrently "
                "and results are returned together. Use this when you have multiple "
                "independent subtasks."
            ),
            parameters=[
                ToolParameter(
                    name="tasks",
                    type="array",
                    description="List of task descriptions to run in parallel",
                ),
            ],
        )

        # Limit concurrent subagent API calls to avoid rate limits / failover state corruption
        _batch_semaphore = asyncio.Semaphore(3)

        async def batch_executor(args: dict[str, Any]) -> ToolResult:
            tasks = args.get("tasks", [])
            if not tasks or not isinstance(tasks, list):
                return ToolResult(tool_use_id="", content="Error: tasks must be a non-empty list", is_error=True)

            async def run_one(task_desc: str, idx: int) -> str:
                async with _batch_semaphore:
                    can, reason = agent_ref._subagent_registry.can_spawn(0, "main")
                    if not can:
                        return f"[Task {idx+1}: SKIPPED] {reason}"
                    entry = agent_ref._subagent_registry.spawn(
                        parent_session_key="main", task=task_desc, depth=0,
                    )
                    agent_ref._subagent_registry.mark_running(entry.id)
                    try:
                        result = await agent_ref.run(task_desc, session_id=entry.session_key)
                        agent_ref._subagent_registry.mark_completed(
                            entry.id, text=result.text, error=result.error
                        )
                        output = result.text or ""
                        if result.error:
                            output += f"\nError: {result.error}"
                        return f"[Task {idx+1}: {entry.id}]\n{output}"
                    except Exception as e:
                        agent_ref._subagent_registry.mark_completed(entry.id, error=str(e))
                        return f"[Task {idx+1}: FAILED] {e}"

            results = await asyncio.gather(
                *[run_one(t, i) for i, t in enumerate(tasks)],
                return_exceptions=True,
            )

            output_parts = []
            for r in results:
                if isinstance(r, Exception):
                    output_parts.append(f"[FAILED] {r}")
                else:
                    output_parts.append(str(r))

            return ToolResult(tool_use_id="", content="\n\n---\n\n".join(output_parts))

        self.tool_registry.register(batch_spawn_def, batch_executor, group="subagent")

    async def _initial_memory_index(self) -> None:
        """Index existing memory files on first run."""
        searcher = self._get_memory_searcher()
        if not searcher:
            return

        import logging
        log = logging.getLogger("openclaw.memory")
        indexed = 0

        # Workspace MEMORY.md
        workspace_memory = self.workspace / "MEMORY.md"
        if workspace_memory.exists():
            try:
                indexed += await searcher.index_file(workspace_memory)
            except Exception:
                pass

        # memory/ directory .md files
        memory_dir = self.config.memory.resolved_dir
        if memory_dir.is_dir():
            for md_file in sorted(memory_dir.glob("*.md")):
                try:
                    indexed += await searcher.index_file(md_file)
                except Exception:
                    pass

        if indexed > 0:
            log.info("Initial memory index: %d chunks indexed", indexed)

    async def _index_previous_sessions(self, current_session_id: str) -> None:
        """Index past session JSONL files into memory (excluding current)."""
        searcher = self._get_memory_searcher()
        if not searcher:
            return

        session_dir = self.config.session.resolved_dir
        if not session_dir.is_dir():
            return

        for jsonl_file in session_dir.glob("*.jsonl"):
            if jsonl_file.stem == current_session_id:
                continue
            try:
                await searcher.index_session_jsonl(jsonl_file)
            except Exception:
                pass

    async def run(
        self,
        message: str,
        *,
        session_id: str = "default",
        thinking: ThinkingLevel | None = None,
    ) -> RunResult:
        """Run the agent with a user message."""
        await self._ensure_initialized(session_id)
        ctx = self._build_context(session_id, thinking)
        return await run(ctx, message)

    async def _ensure_initialized(self, session_id: str) -> None:
        """One-time initialization: memory index + session index + curation."""
        if not getattr(self, "_memory_indexed", False):
            await self._initial_memory_index()
            self._memory_indexed = True
        if not getattr(self, "_sessions_indexed", False):
            await self._index_previous_sessions(session_id)
            self._sessions_indexed = True
        if not getattr(self, "_curation_checked", False):
            await self._check_memory_curation()
            self._curation_checked = True

    async def _check_memory_curation(self) -> None:
        """Check if memory curation should run (daily pattern promotion)."""
        try:
            from openclaw.memory.curation import curate_memories
            memory_dir = self.config.memory.resolved_dir
            if memory_dir.is_dir():
                await curate_memories(
                    memory_dir,
                    str(self.workspace),
                    self.provider,
                    self.config.models.default,
                    embedding_provider=self._embedding_provider,
                )
        except Exception:
            pass

    async def stream(
        self,
        message: str,
        *,
        session_id: str = "default",
        thinking: ThinkingLevel | None = None,
    ):
        """Stream the agent response."""
        await self._ensure_initialized(session_id)
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
