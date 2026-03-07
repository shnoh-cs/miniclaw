"""Cache-TTL based session pruning (in-memory only, no disk modification)."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from openclaw.agent.types import (
    AgentMessage,
    ImageBlock,
    TextBlock,
    ToolResultBlock,
)

if TYPE_CHECKING:
    from openclaw.config import PruningConfig


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


def _total_tool_result_chars(messages: list[AgentMessage]) -> int:
    """Calculate total characters in tool result blocks."""
    total = 0
    for msg in messages:
        for block in msg.content:
            if isinstance(block, ToolResultBlock):
                total += len(block.content)
    return total


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


def prune_messages(
    messages: list[AgentMessage],
    config: PruningConfig,
    state: PruningState,
) -> list[AgentMessage]:
    """Prune old tool results from messages (in-memory only).

    This does NOT modify the on-disk session file.

    Two-tier strategy:
    1. Soft-trim: truncate large tool results to head+tail
    2. Hard-clear: replace entire results with placeholder

    Only activates when:
    - mode is "cache-ttl"
    - TTL has expired since last API call
    - Total prunable tool content exceeds minimum threshold
    """
    if config.mode != "cache-ttl":
        return messages

    if not state.is_ttl_expired(config.ttl_seconds):
        return messages  # cache still fresh

    total_tool_chars = _total_tool_result_chars(messages)
    if total_tool_chars < config.min_prunable_tool_chars:
        return messages  # not enough to prune

    # Protect last N assistant messages and their associated tool results
    protected_indices: set[int] = set()
    assistant_count = 0
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].role == "assistant":
            assistant_count += 1
            protected_indices.add(i)
            # Also protect the next message (tool results)
            if i + 1 < len(messages):
                protected_indices.add(i + 1)
            if assistant_count >= config.keep_last_assistants:
                break

    # Calculate context budget for hard-clear threshold
    context_chars = sum(
        len(b.content) if isinstance(b, ToolResultBlock) else 0
        for msg in messages
        for b in msg.content
    )
    hard_clear_threshold = int(context_chars * config.hard_clear_ratio)

    pruned: list[AgentMessage] = []
    for i, msg in enumerate(messages):
        if i in protected_indices or _has_images(msg):
            pruned.append(msg)
            continue

        new_blocks = []
        for block in msg.content:
            if not isinstance(block, ToolResultBlock):
                new_blocks.append(block)
                continue

            content_len = len(block.content)
            if content_len <= config.soft_trim_chars:
                new_blocks.append(block)
            elif content_len > hard_clear_threshold:
                new_blocks.append(_hard_clear_result(block))
            else:
                new_blocks.append(_soft_trim_result(block))

        pruned.append(msg.model_copy(update={"content": new_blocks}))

    return pruned
