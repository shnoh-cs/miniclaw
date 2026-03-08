"""Cache-TTL based session pruning (in-memory only, no disk modification)."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from openclaw.agent.types import (
    AgentMessage,
    ImageBlock,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
)

if TYPE_CHECKING:
    from openclaw.config import PruningConfig

# Approximate char cost for an image block in context budget estimation.
IMAGE_CHAR_ESTIMATE = 8_000

# Ratio-based gate: pruning activates when context utilization exceeds this fraction
# of the context window (chars = context_window_tokens * CHARS_PER_TOKEN_ESTIMATE).
CHARS_PER_TOKEN_ESTIMATE = 4
DEFAULT_SOFT_TRIM_RATIO = 0.3
DEFAULT_HARD_CLEAR_RATIO = 0.5


class PruningState:
    """Tracks the last API call time for TTL-based cache pruning."""

    def __init__(self) -> None:
        self.last_api_call_time: float = 0.0

    def touch(self) -> None:
        self.last_api_call_time = time.time()

    def is_ttl_expired(self, ttl_seconds: int) -> bool:
        if self.last_api_call_time == 0.0:
            return True  # first call
        return (time.time() - self.last_api_call_time) > ttl_seconds


def _has_images(msg: AgentMessage) -> bool:
    """Check if a message contains image blocks (never pruned)."""
    return any(isinstance(b, ImageBlock) for b in msg.content)


def _estimate_message_chars(msg: AgentMessage) -> int:
    """Estimate the character cost of a message, counting images as IMAGE_CHAR_ESTIMATE."""
    total = 0
    for block in msg.content:
        if isinstance(block, ToolResultBlock):
            total += len(block.content)
        elif isinstance(block, ImageBlock):
            total += IMAGE_CHAR_ESTIMATE
        elif isinstance(block, TextBlock):
            total += len(block.text)
        else:
            # ToolUseBlock — rough estimate of serialized arguments.
            try:
                import json
                total += len(json.dumps(block.input))  # type: ignore[union-attr]
            except Exception:
                total += 128
    return total


def _estimate_context_chars(messages: list[AgentMessage]) -> int:
    """Total estimated chars across all messages (images counted at IMAGE_CHAR_ESTIMATE)."""
    return sum(_estimate_message_chars(m) for m in messages)


def _soft_trim_result(block: ToolResultBlock, head: int = 1500, tail: int = 1500) -> ToolResultBlock:
    """Soft-trim: preserve head + tail with ellipsis."""
    content = block.content
    if len(content) <= head + tail + 100:
        return block
    trimmed = (
        content[:head]
        + f"\n\n... [{len(content)} chars, soft-trimmed] ...\n\n"
        + content[-tail:]
    )
    return ToolResultBlock(
        tool_use_id=block.tool_use_id,
        content=trimmed,
        is_error=block.is_error,
    )


def _hard_clear_result(block: ToolResultBlock) -> ToolResultBlock:
    """Hard-clear: replace entire content with placeholder."""
    return ToolResultBlock(
        tool_use_id=block.tool_use_id,
        content="[Old tool result content cleared]",
        is_error=block.is_error,
    )


IMAGE_PRUNED_PLACEHOLDER = "[image data removed — already processed by model]"


def prune_processed_images(
    messages: list[AgentMessage],
    *,
    keep_last_turns: int = 2,
) -> list[AgentMessage]:
    """Replace ImageBlock instances in older turns with a text placeholder.

    Images consume significant context (base64 data).  Once the model has
    already seen an image and produced a response, the raw image data is no
    longer needed.  This function walks through messages and replaces
    ``ImageBlock`` instances with a lightweight ``TextBlock`` placeholder,
    *except* for images in the last ``keep_last_turns`` user+assistant turn
    pairs (counted from the end of the conversation).

    The replacement is done on a shallow copy — the original list is not
    modified.
    """
    if keep_last_turns < 0:
        keep_last_turns = 0

    # Identify the protected tail: last N turn-pairs (user + assistant).
    # We count assistant messages from the end and protect everything from
    # the Nth-from-last assistant onward.
    protected_start = len(messages)
    if keep_last_turns > 0:
        remaining = keep_last_turns
        for i in range(len(messages) - 1, -1, -1):
            if messages[i].role in ("assistant", "user"):
                remaining -= 1
                if remaining <= 0:
                    protected_start = i
                    break

    result: list[AgentMessage] | None = None

    for i in range(0, min(protected_start, len(messages))):
        msg = messages[i]
        if not any(isinstance(b, ImageBlock) for b in msg.content):
            continue

        new_blocks: list = []
        changed = False
        for block in msg.content:
            if isinstance(block, ImageBlock):
                new_blocks.append(TextBlock(text=IMAGE_PRUNED_PLACEHOLDER))
                changed = True
            else:
                new_blocks.append(block)

        if changed:
            if result is None:
                result = list(messages)
            result[i] = msg.model_copy(update={"content": new_blocks})

    return result if result is not None else messages


def _find_first_user_index(messages: list[AgentMessage]) -> int | None:
    """Return the index of the first user message, or None if there are no user messages."""
    for i, msg in enumerate(messages):
        if msg.role == "user":
            return i
    return None


def _find_assistant_cutoff_index(
    messages: list[AgentMessage],
    keep_last_assistants: int,
) -> int | None:
    """Find the index of the Nth-from-last assistant message.

    Everything from this index onward (a contiguous protected tail) is shielded
    from pruning. Returns None when there are not enough assistant messages to
    establish a protected tail.
    """
    if keep_last_assistants <= 0:
        return len(messages)

    remaining = keep_last_assistants
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].role != "assistant":
            continue
        remaining -= 1
        if remaining == 0:
            return i

    # Not enough assistant messages to establish a protected tail.
    return None


def prune_messages(
    messages: list[AgentMessage],
    config: PruningConfig,
    state: PruningState,
    *,
    context_window_tokens: int = 0,
    prunable_tools: set[str] | None = None,
) -> list[AgentMessage]:
    """Prune old tool results from messages (in-memory only).

    This does NOT modify the on-disk session file.

    Two-tier strategy:
    1. Soft-trim: truncate large tool results to head+tail
    2. Hard-clear: iteratively replace oldest prunable results with placeholder,
       rechecking ratio after each, stopping once under threshold

    Only activates when:
    - mode is "cache-ttl"
    - TTL has expired since last API call
    - Context utilization ratio exceeds softTrimRatio (default 0.3)
    - Total prunable tool content exceeds minimum threshold

    Parameters
    ----------
    context_window_tokens:
        The model's context window size in tokens.  When >0 the pruning gate
        uses ratio-based thresholds (softTrimRatio / hardClearRatio) against
        ``context_window_tokens * CHARS_PER_TOKEN_ESTIMATE``.  Falls back to
        the legacy absolute-char mode when 0.
    prunable_tools:
        If provided, only prune tool results whose originating tool name is in
        this set.  Requires resolving ``tool_use_id`` → tool name via
        ``ToolUseBlock`` in assistant messages.  If ``None``, all tool results
        are eligible for pruning (current default behavior).
    """
    if config.mode != "cache-ttl":
        return messages

    if not state.is_ttl_expired(config.ttl_seconds):
        return messages  # cache still fresh

    # Compute char window for ratio-based gating.
    if context_window_tokens > 0:
        char_window = context_window_tokens * CHARS_PER_TOKEN_ESTIMATE
    else:
        # Legacy fallback: derive from config.  Use a generous estimate so the
        # ratio-based gate still works.
        char_window = config.soft_trim_chars * 10 if config.soft_trim_chars > 0 else 40_000

    total_chars = _estimate_context_chars(messages)
    ratio = total_chars / char_window if char_window > 0 else 1.0

    soft_trim_ratio = DEFAULT_SOFT_TRIM_RATIO
    hard_clear_ratio = getattr(config, "hard_clear_ratio", DEFAULT_HARD_CLEAR_RATIO)

    if ratio < soft_trim_ratio:
        return messages  # not enough context utilization to warrant pruning

    # Bootstrap protection: never prune anything before the first user message.
    # This protects initial "identity" reads (SOUL.md, USER.md, etc.) which
    # typically happen before the first inbound user message.
    first_user_index = _find_first_user_index(messages)
    prune_start = first_user_index if first_user_index is not None else len(messages)

    # Protected tail: contiguous tail from the Nth-from-last assistant onward.
    cutoff_index = _find_assistant_cutoff_index(messages, config.keep_last_assistants)
    if cutoff_index is None:
        return messages  # not enough assistant messages to establish protected tail

    # Build tool_use_id → tool_name map when prunable_tools filtering is active.
    _tool_id_to_name: dict[str, str] | None = None
    if prunable_tools is not None:
        _tool_id_to_name = {}
        for msg in messages:
            if msg.role == "assistant":
                for block in msg.content:
                    if isinstance(block, ToolUseBlock):
                        _tool_id_to_name[block.id] = block.name

    # Collect prunable tool result indices.
    prunable_indices: list[int] = []

    # Build soft-trimmed output.
    result: list[AgentMessage] | None = None

    for i in range(prune_start, cutoff_index):
        msg = messages[i]
        if _has_images(msg):
            continue

        has_tool_results = any(isinstance(b, ToolResultBlock) for b in msg.content)
        if not has_tool_results:
            continue

        # If prunable_tools is set, skip messages where none of the tool
        # results belong to the allowed set.
        if _tool_id_to_name is not None and prunable_tools is not None:
            if not any(
                isinstance(b, ToolResultBlock)
                and _tool_id_to_name.get(b.tool_use_id, "") in prunable_tools
                for b in msg.content
            ):
                continue

        prunable_indices.append(i)

        new_blocks = []
        changed = False
        for block in msg.content:
            if not isinstance(block, ToolResultBlock):
                new_blocks.append(block)
                continue

            content_len = len(block.content)
            if content_len > config.soft_trim_chars:
                trimmed = _soft_trim_result(block)
                new_blocks.append(trimmed)
                before_chars = content_len
                after_chars = len(trimmed.content)
                total_chars += after_chars - before_chars
                changed = True
            else:
                new_blocks.append(block)

        if changed:
            if result is None:
                result = list(messages)
            result[i] = msg.model_copy(update={"content": new_blocks})

    output_after_soft = result if result is not None else messages

    # Recheck ratio after soft trim.
    ratio = total_chars / char_window if char_window > 0 else 1.0
    if ratio < hard_clear_ratio:
        return output_after_soft

    # Check minimum prunable tool chars.
    prunable_tool_chars = 0
    for i in prunable_indices:
        m = output_after_soft[i]
        prunable_tool_chars += _estimate_message_chars(m)
    if prunable_tool_chars < config.min_prunable_tool_chars:
        return output_after_soft

    # Iterative hard clear: clear one prunable result at a time, rechecking
    # ratio after each, stopping once under threshold.
    for i in prunable_indices:
        if ratio < hard_clear_ratio:
            break

        source_msg = output_after_soft[i]

        new_blocks = []
        changed = False
        for block in source_msg.content:
            if not isinstance(block, ToolResultBlock):
                new_blocks.append(block)
                continue

            before_chars = len(block.content)
            cleared = _hard_clear_result(block)
            after_chars = len(cleared.content)
            if after_chars < before_chars:
                new_blocks.append(cleared)
                total_chars += after_chars - before_chars
                changed = True
            else:
                new_blocks.append(block)

        if changed:
            if result is None:
                result = list(messages)
            result[i] = source_msg.model_copy(update={"content": new_blocks})
            # Update reference for subsequent iterations.
            output_after_soft = result  # type: ignore[assignment]

        ratio = total_chars / char_window if char_window > 0 else 1.0

    return result if result is not None else messages
