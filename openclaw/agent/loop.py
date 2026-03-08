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
import json
import re
import time
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

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
        flush_result = await execute_memory_flush(
            ctx.session, ctx.provider, ctx.workspace_dir
        )
        # Memory flush is silent — don't show to user

    # Sanitize user input
    clean_input = sanitize_text(user_input)

    # Build user message
    user_blocks: list[ContentBlock] = [TextBlock(text=clean_input)]
    user_msg = AgentMessage(role="user", content=user_blocks)
    ctx.session.append(user_msg)

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

        if status.action == ContextAction.COMPACT:
            entry = await compact_session(
                ctx.session, ctx.provider, ctx.config.compaction,
                ctx.config.context.max_tokens,
            )
            if entry:
                result.compacted = True

        if status.action == ContextAction.ERROR:
            # Overflow — force compaction and retry
            entry = await compact_session(
                ctx.session, ctx.provider, ctx.config.compaction,
                ctx.config.context.max_tokens,
            )
            if entry:
                result.compacted = True
            else:
                result.error = "Context overflow: unable to compact"
                break

        # Apply pruning (in-memory only)
        pruned_messages = prune_messages(
            ctx.session.messages, ctx.config.pruning, ctx.pruning_state
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
