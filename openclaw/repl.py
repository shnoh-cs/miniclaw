"""Interactive REPL for the OpenClaw agent."""

from __future__ import annotations

import asyncio
import json
import sys

from openclaw.agent.api import Agent
from openclaw.agent.loop import run
from openclaw.agent.types import ToolResult


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
    console.print("Type /quit to exit, /new for new session, /context for diagnosis\n")

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

            if user_input.strip() == "/context":
                from openclaw.context.diagnosis import diagnose_context
                from openclaw.prompt.builder import build_system_prompt
                ctx = agent._build_context(session_id)
                ctx.session.load()
                tool_defs = ctx.tool_registry.get_definitions()
                sys_prompt = build_system_prompt(
                    config=agent.config,
                    tools=tool_defs,
                    workspace_dir=str(agent.workspace),
                )
                diag = diagnose_context(
                    ctx.session.messages,
                    sys_prompt,
                    tool_defs,
                    agent.config.context.max_tokens,
                    compaction_summary=ctx.session.latest_compaction_summary,
                )
                console.print(f"[dim]{diag.format()}[/dim]")
                continue

            if user_input.strip() == "/compact":
                console.print("[dim]Compacting session...[/dim]")
                ctx = agent._build_context(session_id)
                ctx.session.load()
                from openclaw.session.compaction import compact_session
                entry = await compact_session(
                    ctx.session, ctx.provider, agent.config.compaction,
                    agent.config.context.max_tokens,
                    workspace_dir=str(agent.workspace),
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
