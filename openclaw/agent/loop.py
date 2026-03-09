"""Core agent loop: run → attempt → stream → tool dispatch → re-entry.

This is the heart of the agent harness. It orchestrates:
1. Session loading and context management
2. System prompt assembly
3. Model streaming with tool calling
4. Tool execution and result handling
5. Compaction and pruning
6. Multi-turn re-entry (loop until no more tool calls)
"""

from __future__ import annotations

import asyncio
import datetime
import json
import re
import time
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from openclaw.memory.search import MemorySearcher

from openclaw.agent.types import (
    AgentMessage,
    ContentBlock,
    FailoverReason,
    RunResult,
    TextBlock,
    ThinkingLevel,
    TokenUsage,
    ToolDefinition,
    ToolResult,
    ToolResultBlock,
    ToolUseBlock,
)
from openclaw.config import AppConfig
from openclaw.hooks import HookRunner
from openclaw.context.guard import ContextAction, ContextGuard
from openclaw.model.failover import FailoverManager, classify_error, should_failover
from openclaw.model.provider import ModelProvider, StreamChunk, parse_tool_calls_from_text
from openclaw.model.thinking import resolve_thinking
from openclaw.prompt.bootstrap import BootstrapContext, load_bootstrap_files
from openclaw.prompt.builder import build_system_prompt
from openclaw.prompt.sanitize import sanitize_text
from openclaw.session.compaction import compact_session
from openclaw.session.manager import SessionManager, SessionWriteLock
from openclaw.session.memory_flush import execute_memory_flush, should_flush
from openclaw.session.pruning import PruningState, prune_messages, prune_processed_images
from openclaw.skills.loader import build_skills_prompt, load_skills
from openclaw.tools.registry import (
    ToolLoopDetector,
    ToolRegistry,
    cap_tool_result_for_session,
    synthesize_missing_tool_result,
    truncate_tool_result,
)

# Thinking/final tag patterns
_THINKING_PATTERN = re.compile(r"<thinking>(.*?)</thinking>", re.DOTALL)
_FINAL_PATTERN = re.compile(r"<final>(.*?)</final>", re.DOTALL)

# Thinking-related error patterns (model rejects the requested thinking level)
_THINKING_ERROR_PATTERN = re.compile(
    r"thinking|reasoning|extended.?thinking|budget_tokens|not.?supported.*think",
    re.IGNORECASE,
)

# NO_REPLY suppression token
NO_REPLY = "NO_REPLY"

# Max turns per run to prevent infinite loops
MAX_TURNS_PER_RUN = 50


@dataclass
class AgentContext:
    """Runtime context for the agent loop."""

    config: AppConfig
    provider: ModelProvider
    session: SessionManager
    tool_registry: ToolRegistry
    context_guard: ContextGuard
    failover: FailoverManager = field(default_factory=FailoverManager)
    loop_detector: ToolLoopDetector = field(default_factory=ToolLoopDetector)
    pruning_state: PruningState = field(default_factory=PruningState)
    thinking: ThinkingLevel = ThinkingLevel.OFF
    workspace_dir: str = ""
    bootstrap_ctx: BootstrapContext | None = None
    skills_prompt: str = ""
    model: str = ""
    on_stream: Callable[[str], None] | None = None  # streaming text callback
    on_thinking: Callable[[str], None] | None = None  # thinking callback
    on_tool_start: Callable[[str, dict], None] | None = None  # tool start callback
    on_tool_end: Callable[[str, ToolResult], None] | None = None  # tool end callback
    hook_runner: HookRunner | None = None
    memory_searcher: MemorySearcher | None = None  # for post-flush re-indexing
    auto_recall_context: str = ""  # auto-recalled memory snippets injected per turn
    recovery_checkpoint: str = ""  # post-compaction checkpoint for context recovery


async def _run_flush_with_tools(ctx: AgentContext, model: str) -> str | None:
    """Run memory flush through agent loop with tool access.

    Unlike plain text flush, this lets the model:
    1. Search existing memories (memory_search) to avoid duplicates
    2. Save structured memories (memory_save) with proper formatting
    Falls back to plain completion on failure.
    """
    import tempfile

    # Build conversation summary for flush context
    recent = ctx.session.messages[-20:]
    conv_lines = []
    for m in recent:
        text = m.text[:500]
        if text:
            conv_lines.append(f"[{m.role}] {text}")
    conv_summary = "\n".join(conv_lines)

    date_str = datetime.date.today().isoformat()
    flush_prompt = (
        f"Store durable memories now. Recent conversation:\n\n"
        f"{conv_summary}\n\n"
        f"Write any lasting notes (facts, decisions, user preferences) to "
        f"memory/{date_str}.md using memory_save.\n"
        f"Reply with NO_REPLY if nothing important to save."
    )

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            flush_session = SessionManager(Path(tmpdir), f"flush-{int(time.time())}")
            flush_session._loaded = True

            # Add flush prompt as user message
            flush_user = AgentMessage(
                role="user", content=[TextBlock(text=flush_prompt)]
            )
            flush_session.messages.append(flush_user)

            flush_ctx = replace(
                ctx,
                session=flush_session,
                bootstrap_ctx=None,
                skills_prompt="",
                loop_detector=ToolLoopDetector(),
                pruning_state=PruningState(),
                on_stream=None,
                on_thinking=None,
                on_tool_start=None,
                on_tool_end=None,
                auto_recall_context="",
                recovery_checkpoint="",
            )

            result = await _attempt_loop(flush_ctx, model)
            text = result.text or ""
            if text.strip().upper() == NO_REPLY:
                return None
            return text
    except Exception:
        # Fallback: plain completion (no tools)
        return await execute_memory_flush(
            ctx.session, ctx.provider, ctx.workspace_dir
        )


def _load_recovery_checkpoint(ctx: AgentContext) -> None:
    """Load .context-checkpoint.md after compaction for context recovery."""
    if not ctx.workspace_dir:
        return
    checkpoint_path = Path(ctx.workspace_dir) / ".context-checkpoint.md"
    if not checkpoint_path.exists():
        return
    try:
        ctx.recovery_checkpoint = checkpoint_path.read_text(
            encoding="utf-8", errors="replace"
        )[:2000]
    except Exception:
        pass


async def run(
    ctx: AgentContext,
    user_input: str,
    images: list[str] | None = None,
) -> RunResult:
    """Execute the full agent loop for a user message.

    This is the main entry point. It:
    1. Loads session and checks context
    2. Appends user message
    3. Runs attempt(s) with tool calling loop
    4. Handles compaction if needed
    5. Returns the final result
    """
    result = RunResult()
    model = ctx.model or ctx.config.models.default

    # Load session
    ctx.session.load()

    # Check for memory flush before potential compaction
    if await should_flush(ctx.session, ctx.config.compaction, ctx.config.context.max_tokens):
        await _run_flush_with_tools(ctx, model)

    # Sanitize user input
    clean_input = sanitize_text(user_input)

    # Build user message
    user_blocks: list[ContentBlock] = [TextBlock(text=clean_input)]
    user_msg = AgentMessage(role="user", content=user_blocks)
    ctx.session.append(user_msg)

    # Session delta sync: re-index current session JSONL if it has grown enough
    if ctx.memory_searcher:
        try:
            jsonl_path = ctx.session.session_dir / f"{ctx.session.session_id}.jsonl"
            if jsonl_path.exists():
                await ctx.memory_searcher.sync_session_if_needed(jsonl_path)
        except Exception:
            pass

    # Auto-recall: scope-aware memory search (long-term / episodic / session)
    if ctx.memory_searcher and len(clean_input) >= 10:
        try:
            recalls = await ctx.memory_searcher.search(
                clean_input[:500], max_results=5
            )
            if recalls:
                long_term: list[str] = []  # MEMORY.md
                episodic: list[str] = []   # daily notes
                for r in recalls:
                    if r.final_score < 0.3:
                        continue
                    path = r.chunk.file_path
                    # Session chunks are already in context — skip
                    if r.chunk.source_type == "session":
                        continue
                    snippet = r.chunk.text[:300]
                    if "MEMORY.md" in path:
                        # Boost long-term memories
                        score = min(r.final_score * 1.2, 1.0)
                        long_term.append(f"[{score:.2f}] {snippet}")
                    else:
                        episodic.append(
                            f"[{r.final_score:.2f}] {path}: {snippet}"
                        )

                recall_sections: list[str] = []
                if long_term:
                    recall_sections.append(
                        "## Long-term Memory\n" + "\n\n".join(long_term[:2])
                    )
                if episodic:
                    recall_sections.append(
                        "## Recent Context\n" + "\n\n".join(episodic[:2])
                    )
                if recall_sections:
                    ctx.auto_recall_context = "\n\n".join(recall_sections)
        except Exception:
            pass

    # Fire pre_message hook
    if ctx.hook_runner:
        await ctx.hook_runner.fire("pre_message")

    # Run the attempt loop
    try:
        result = await _attempt_loop(ctx, model)
    except Exception as e:
        reason = classify_error(e)
        if ctx.hook_runner:
            await ctx.hook_runner.fire(
                "on_error",
                error_type=type(e).__name__,
                error_message=str(e),
            )
        if should_failover(reason):
            reason, next_model = ctx.failover.handle_error(e)
            if next_model:
                result = await _attempt_loop(ctx, next_model)
            else:
                result.error = f"All models/profiles exhausted. Last error: {e}"
        else:
            result.error = str(e)

    # Fire post_message hook
    if ctx.hook_runner:
        await ctx.hook_runner.fire(
            "post_message",
            text_length=len(result.text or ""),
            tool_count=result.tool_calls_count,
        )

    return result


async def _attempt_loop(ctx: AgentContext, model: str) -> RunResult:
    """Run the core attempt loop: model call → tool execution → repeat."""
    result = RunResult()
    turn_count = 0

    while turn_count < MAX_TURNS_PER_RUN:
        turn_count += 1

        # Check context budget
        estimated_tokens = ctx.session.estimate_tokens()
        status = ctx.context_guard.check(estimated_tokens)

        # Auto-adjust config at high utilization thresholds
        if status.utilization >= 0.70:
            from openclaw.context.diagnosis import diagnose_context
            diag = diagnose_context(
                ctx.session.messages, "", max_tokens=ctx.config.context.max_tokens
            )
            diag.apply_adjustments(ctx.config.context)

        if status.action == ContextAction.COMPACT:
            entry = await compact_session(
                ctx.session, ctx.provider, ctx.config.compaction,
                ctx.config.context.max_tokens,
                workspace_dir=ctx.workspace_dir,
                reserve_tokens_floor=ctx.config.context.reserve_tokens_floor,
            )
            if entry:
                result.compacted = True
                _load_recovery_checkpoint(ctx)

        if status.action == ContextAction.ERROR:
            # Overflow — force compaction and retry
            entry = await compact_session(
                ctx.session, ctx.provider, ctx.config.compaction,
                ctx.config.context.max_tokens,
                workspace_dir=ctx.workspace_dir,
                reserve_tokens_floor=ctx.config.context.reserve_tokens_floor,
            )
            if entry:
                result.compacted = True
                _load_recovery_checkpoint(ctx)
            else:
                result.error = "Context overflow: unable to compact"
                break

        # Apply pruning (in-memory only) — only prune large-output tools
        _PRUNABLE_TOOLS = {"bash", "read", "web_fetch", "pdf", "hancom", "process", "image"}
        pruned_messages = prune_messages(
            ctx.session.messages, ctx.config.pruning, ctx.pruning_state,
            context_window_tokens=ctx.config.context.max_tokens,
            prunable_tools=_PRUNABLE_TOOLS,
        )

        # Strip already-processed images from older turns
        pruned_messages = prune_processed_images(pruned_messages)

        # Build system prompt
        system_prompt = build_system_prompt(
            config=ctx.config,
            tools=ctx.tool_registry.get_definitions(),
            skills_prompt=ctx.skills_prompt,
            bootstrap_ctx=ctx.bootstrap_ctx,
            thinking=ctx.thinking,
            model=model,
            workspace_dir=ctx.workspace_dir,
            compaction_summary=ctx.session.latest_compaction_summary,
        )

        # Inject auto-recalled memories
        if ctx.auto_recall_context:
            system_prompt += f"\n\n{ctx.auto_recall_context}"

        # Inject recovery checkpoint after compaction
        if ctx.recovery_checkpoint:
            system_prompt += (
                "\n\n## Recovery Checkpoint (post-compaction)\n"
                "The following is the last working state before compaction:\n\n"
                f"{ctx.recovery_checkpoint}"
            )

        # Stream model response
        accumulated_text = ""
        accumulated_thinking = ""
        tool_calls: list[ToolUseBlock] = []
        partial_tool_args: dict[int, dict[str, str]] = {}  # index → {name, args_buffer}
        usage = TokenUsage()

        try:
            async for chunk in ctx.provider.stream(
                system=system_prompt,
                messages=pruned_messages,
                tools=ctx.tool_registry.get_definitions(),
                model=model,
                thinking=ctx.thinking,
            ):
                # Handle text
                if chunk.text:
                    accumulated_text += chunk.text
                    if ctx.on_stream:
                        ctx.on_stream(chunk.text)

                # Handle native tool calls (streaming accumulation)
                if chunk.tool_calls:
                    tool_calls.extend(chunk.tool_calls)

                # Handle usage (last chunk)
                if chunk.usage:
                    usage = chunk.usage
        except Exception as stream_err:
            # Thinking level fallback: if the model rejects the thinking
            # level, drop one level and retry the same turn without
            # counting it as a failover attempt.
            if (
                ctx.thinking != ThinkingLevel.OFF
                and _is_thinking_error(stream_err)
            ):
                lower = ctx.thinking.fallback()
                if lower != ctx.thinking:
                    ctx.thinking = lower
                    turn_count -= 1  # don't count this turn
                    continue
            raise

        # Update pruning state
        ctx.pruning_state.touch()

        # Parse thinking tags from text
        thinking_text, final_text = _parse_thinking(accumulated_text)
        if thinking_text and ctx.on_thinking:
            ctx.on_thinking(thinking_text)
        if final_text:
            accumulated_text = final_text

        # Parse prompt-based tool calls if no native calls
        if not tool_calls:
            cleaned, parsed_calls = parse_tool_calls_from_text(accumulated_text)
            if parsed_calls:
                tool_calls = parsed_calls
                accumulated_text = cleaned

        # Check NO_REPLY
        if accumulated_text.strip().upper() == NO_REPLY:
            accumulated_text = ""

        # Build assistant message
        assistant_blocks: list[ContentBlock] = []
        if accumulated_text.strip():
            assistant_blocks.append(TextBlock(text=accumulated_text.strip()))
        for tc in tool_calls:
            assistant_blocks.append(tc)

        if assistant_blocks:
            assistant_msg = AgentMessage(role="assistant", content=assistant_blocks)
            ctx.session.append(assistant_msg)

        result.usage = TokenUsage(
            input_tokens=result.usage.input_tokens + usage.input_tokens,
            output_tokens=result.usage.output_tokens + usage.output_tokens,
        )

        # If no tool calls, we're done
        if not tool_calls:
            result.text = accumulated_text.strip()
            ctx.failover.mark_success()
            break

        # Execute tool calls
        tool_result_blocks: list[ContentBlock] = []

        for tc in tool_calls:
            result.tool_calls_count += 1

            # Notify callback
            if ctx.on_tool_start:
                ctx.on_tool_start(tc.name, tc.input)

            # Fire pre_tool_call hook
            if ctx.hook_runner:
                await ctx.hook_runner.fire(
                    "pre_tool_call",
                    tool_name=tc.name,
                    tool_args=json.dumps(tc.input),
                )

            # Execute
            t0 = time.monotonic()
            tool_result = await ctx.tool_registry.execute(tc.name, tc.input)
            tool_result.tool_use_id = tc.id
            tool_duration = round(time.monotonic() - t0, 3)

            # Guard: truncate oversized results
            max_chars = ctx.context_guard.tool_result_max_chars()
            tool_result.content = truncate_tool_result(tool_result.content, max_chars)

            # Session guard: hard cap before session write
            tool_result.content = cap_tool_result_for_session(tool_result.content)

            # Loop detection
            warning = ctx.loop_detector.record(tc.name, tc.input, tool_result.content)
            if warning:
                if "CRITICAL" in warning:
                    tool_result.content += f"\n\n⚠️ {warning}"
                    tool_result.is_error = True
                    # Force stop after critical
                    if ctx.on_tool_end:
                        ctx.on_tool_end(tc.name, tool_result)
                    if ctx.hook_runner:
                        await ctx.hook_runner.fire(
                            "post_tool_call",
                            tool_name=tc.name,
                            status="error",
                            duration=tool_duration,
                        )
                    tool_result_blocks.append(
                        ToolResultBlock(
                            tool_use_id=tc.id,
                            content=tool_result.content,
                            is_error=True,
                        )
                    )
                    # Append results and break the tool loop
                    user_result_msg = AgentMessage(role="user", content=tool_result_blocks)
                    ctx.session.append(user_result_msg)
                    result.text = accumulated_text.strip()
                    result.error = warning
                    return result
                else:
                    tool_result.content += f"\n\n⚠️ {warning}"

            if ctx.on_tool_end:
                ctx.on_tool_end(tc.name, tool_result)

            # Fire post_tool_call hook
            if ctx.hook_runner:
                await ctx.hook_runner.fire(
                    "post_tool_call",
                    tool_name=tc.name,
                    status="error" if tool_result.is_error else "ok",
                    duration=tool_duration,
                )

            tool_result_blocks.append(
                ToolResultBlock(
                    tool_use_id=tc.id,
                    content=tool_result.content,
                    is_error=tool_result.is_error,
                )
            )

        # Synthesize missing results for orphaned tool_use blocks.
        # If the model emitted a tool_use that we didn't execute (e.g. unknown
        # tool, or execution was skipped), inject a placeholder so the
        # conversation stays structurally valid for the next model call.
        answered_ids = {
            b.tool_use_id
            for b in tool_result_blocks
            if isinstance(b, ToolResultBlock)
        }
        for tc in tool_calls:
            if tc.id not in answered_ids:
                tool_result_blocks.append(synthesize_missing_tool_result(tc.id))

        # Append tool results as user message (OpenAI convention)
        if tool_result_blocks:
            user_result_msg = AgentMessage(role="user", content=tool_result_blocks)
            ctx.session.append(user_result_msg)

        # Loop continues — model will see tool results and respond

    if turn_count >= MAX_TURNS_PER_RUN:
        result.error = f"Max turns ({MAX_TURNS_PER_RUN}) exceeded"

    result.messages = ctx.session.messages
    return result


def _is_thinking_error(error: Exception) -> bool:
    """Check if an error indicates the model rejected a thinking level.

    Matches messages containing thinking/reasoning-related keywords that
    indicate the requested thinking configuration is unsupported.
    """
    return bool(_THINKING_ERROR_PATTERN.search(str(error)))


def _parse_thinking(text: str) -> tuple[str, str]:
    """Parse <thinking> and <final> tags from model output.

    Returns (thinking_text, final_or_cleaned_text).
    """
    thinking_parts = []
    for match in _THINKING_PATTERN.finditer(text):
        thinking_parts.append(match.group(1).strip())

    # If <final> tag exists, use its content
    final_match = _FINAL_PATTERN.search(text)
    if final_match:
        return "\n".join(thinking_parts), final_match.group(1).strip()

    # Otherwise strip thinking tags and return rest
    cleaned = _THINKING_PATTERN.sub("", text).strip()
    return "\n".join(thinking_parts), cleaned


async def stream_run(
    ctx: AgentContext,
    user_input: str,
) -> AsyncIterator[str]:
    """Streaming version of run() — yields text chunks as they arrive."""
    chunks: list[str] = []

    def on_chunk(text: str) -> None:
        chunks.append(text)

    original_on_stream = ctx.on_stream
    ctx.on_stream = on_chunk

    # Run in background task
    task = asyncio.create_task(run(ctx, user_input))

    # Yield chunks as they accumulate
    yielded = 0
    while not task.done():
        await asyncio.sleep(0.05)
        while yielded < len(chunks):
            yield chunks[yielded]
            yielded += 1

    # Yield remaining chunks
    while yielded < len(chunks):
        yield chunks[yielded]
        yielded += 1

    # Restore
    ctx.on_stream = original_on_stream

    # Propagate exceptions
    task.result()
