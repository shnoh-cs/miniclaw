"""Context window guard: token budget management with overflow recovery."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum, auto
from typing import TYPE_CHECKING

from openclaw.agent.types import AgentMessage, ToolResultBlock

if TYPE_CHECKING:
    from openclaw.config import ContextConfig


class ContextAction(Enum):
    OK = auto()
    COMPACT = auto()
    ERROR = auto()


@dataclass
class ContextStatus:
    action: ContextAction
    current_tokens: int
    max_tokens: int
    utilization: float  # 0.0 to 1.0
    message: str = ""


SAFETY_MARGIN = 1.2  # 20% buffer for estimation inaccuracy

# Hard ceiling for any single tool result regardless of context window size.
HARD_MAX_CHARS = 400_000

# Budget constants (matching the original TypeScript implementation).
CHARS_PER_TOKEN_ESTIMATE = 4
TOOL_RESULT_CHARS_PER_TOKEN_ESTIMATE = 2
CONTEXT_INPUT_HEADROOM_RATIO = 0.75
SINGLE_TOOL_RESULT_CONTEXT_SHARE = 0.5

CONTEXT_LIMIT_TRUNCATION_NOTICE = "[truncated: output exceeded context limit]"
PREEMPTIVE_COMPACTION_PLACEHOLDER = "[compacted: tool output removed to free context]"

# Tail detection: patterns that indicate valuable content near the end.
_IMPORTANT_TAIL_RE = re.compile(
    r"(?:"
    r"error|exception|traceback|panic|fatal"
    r"|summary|result|done|completed|finished|total"
    r"|[\]\}\)]\s*$"  # JSON/bracket close at end of line
    r")",
    re.IGNORECASE,
)

# How many chars from the end to scan for important-tail patterns.
_TAIL_SCAN_CHARS = 2000

# Maximum tail portion even when important tail is detected.
_TAIL_CAP_CHARS = 4000


def _has_important_tail(text: str) -> bool:
    """Detect error/exception/traceback/JSON-close/summary patterns in last 2000 chars."""
    if len(text) <= _TAIL_SCAN_CHARS:
        return False
    tail_region = text[-_TAIL_SCAN_CHARS:]
    return bool(_IMPORTANT_TAIL_RE.search(tail_region))


def _find_newline_cut(text: str, target: int) -> int:
    """Find the nearest newline boundary at or before *target*, staying above 70% of target."""
    if target <= 0 or target >= len(text):
        return target
    newline_pos = text.rfind("\n", 0, target)
    if newline_pos > int(target * 0.7):
        return newline_pos
    return target


def _truncate_text_to_budget(text: str, max_chars: int) -> str:
    """Truncate text to *max_chars* with optional head+tail split.

    - If the tail region contains important patterns, split into head + tail
      (tail capped at _TAIL_CAP_CHARS).
    - Otherwise, head-only with a truncation notice appended.
    - Cuts are aligned to newline boundaries when possible.
    """
    if len(text) <= max_chars:
        return text

    if max_chars <= 0:
        return CONTEXT_LIMIT_TRUNCATION_NOTICE

    suffix = f"\n{CONTEXT_LIMIT_TRUNCATION_NOTICE}"
    body_budget = max(0, max_chars - len(suffix))
    if body_budget <= 0:
        return CONTEXT_LIMIT_TRUNCATION_NOTICE

    if _has_important_tail(text):
        tail_chars = min(_TAIL_CAP_CHARS, body_budget // 3)
        head_chars = body_budget - tail_chars
        head_cut = _find_newline_cut(text, head_chars)
        tail_start = len(text) - tail_chars
        # Align tail start to a newline boundary (search forward).
        nl_after = text.find("\n", tail_start)
        if nl_after != -1 and nl_after < tail_start + 200:
            tail_start = nl_after + 1
        return text[:head_cut] + suffix + "\n" + text[tail_start:]

    # Head-only truncation.
    cut_point = _find_newline_cut(text, body_budget)
    return text[:cut_point] + suffix


class ContextGuard:
    """Manages context window token budget.

    Triggers:
    - COMPACT when utilization exceeds compaction_threshold
    - ERROR when estimated tokens exceed max (after safety margin)
    """

    def __init__(self, config: ContextConfig) -> None:
        self.config = config

    @property
    def effective_max(self) -> int:
        """Max tokens accounting for reserve floor."""
        return self.config.max_tokens - self.config.reserve_tokens_floor

    def check(self, estimated_tokens: int, system_prompt_tokens: int = 0) -> ContextStatus:
        """Check if context is within budget.

        Returns action needed: OK, COMPACT, or ERROR.
        """
        total = int((estimated_tokens + system_prompt_tokens) * SAFETY_MARGIN)
        utilization = total / self.config.max_tokens if self.config.max_tokens > 0 else 1.0

        if total >= self.config.max_tokens:
            return ContextStatus(
                action=ContextAction.ERROR,
                current_tokens=total,
                max_tokens=self.config.max_tokens,
                utilization=utilization,
                message=f"Context overflow: {total} > {self.config.max_tokens} tokens",
            )

        if utilization >= self.config.compaction_threshold:
            return ContextStatus(
                action=ContextAction.COMPACT,
                current_tokens=total,
                max_tokens=self.config.max_tokens,
                utilization=utilization,
                message=f"Context at {utilization:.0%}, compaction recommended",
            )

        return ContextStatus(
            action=ContextAction.OK,
            current_tokens=total,
            max_tokens=self.config.max_tokens,
            utilization=utilization,
        )

    def check_tool_result(self, result_chars: int) -> bool:
        """Check if a single tool result is within the allowed ratio.

        Returns True if OK, False if result should be truncated.
        """
        max_chars = min(self.tool_result_max_chars(), HARD_MAX_CHARS)
        return result_chars <= max_chars

    def tool_result_max_chars(self) -> int:
        """Maximum allowed characters for a single tool result."""
        computed = int(self.config.max_tokens * 4 * self.config.tool_result_max_ratio)
        return min(computed, HARD_MAX_CHARS)

    # ------------------------------------------------------------------
    # Transformer mode: enforce_budget
    # ------------------------------------------------------------------

    def enforce_budget(self, messages: list[AgentMessage]) -> None:
        """Mutate *messages* in-place to fit within the context budget.

        Two passes:
        1. Truncate individual oversized tool results (per-result cap =
           context_window * TOOL_RESULT_CHARS_PER_TOKEN_ESTIMATE * SINGLE_TOOL_RESULT_CONTEXT_SHARE).
        2. Iteratively compact oldest tool results with a placeholder until
           total estimated chars is under budget
           (budget = context_window * CHARS_PER_TOKEN_ESTIMATE * CONTEXT_INPUT_HEADROOM_RATIO).
        """
        context_window = self.config.max_tokens
        if context_window <= 0:
            return

        max_single = max(
            1024,
            int(context_window * TOOL_RESULT_CHARS_PER_TOKEN_ESTIMATE * SINGLE_TOOL_RESULT_CONTEXT_SHARE),
        )
        # Also enforce the hard cap.
        max_single = min(max_single, HARD_MAX_CHARS)

        context_budget_chars = max(
            1024,
            int(context_window * CHARS_PER_TOKEN_ESTIMATE * CONTEXT_INPUT_HEADROOM_RATIO),
        )

        # Pass 1: per-result truncation.
        for msg in messages:
            for idx, block in enumerate(msg.content):
                if not isinstance(block, ToolResultBlock):
                    continue
                if len(block.content) <= max_single:
                    continue
                truncated_text = _truncate_text_to_budget(block.content, max_single)
                msg.content[idx] = ToolResultBlock(
                    tool_use_id=block.tool_use_id,
                    content=truncated_text,
                    is_error=block.is_error,
                )

        # Pass 2: iterative compaction of oldest tool results.
        current_chars = _estimate_messages_chars(messages)
        if current_chars <= context_budget_chars:
            return

        chars_needed = current_chars - context_budget_chars
        reduced = 0
        for msg in messages:
            if reduced >= chars_needed:
                break
            for idx, block in enumerate(msg.content):
                if reduced >= chars_needed:
                    break
                if not isinstance(block, ToolResultBlock):
                    continue
                before = len(block.content)
                if before <= len(PREEMPTIVE_COMPACTION_PLACEHOLDER):
                    continue
                msg.content[idx] = ToolResultBlock(
                    tool_use_id=block.tool_use_id,
                    content=PREEMPTIVE_COMPACTION_PLACEHOLDER,
                    is_error=block.is_error,
                )
                reduced += before - len(PREEMPTIVE_COMPACTION_PLACEHOLDER)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _estimate_messages_chars(messages: list[AgentMessage]) -> int:
    """Quick char-based estimate of total context size."""
    total = 0
    for msg in messages:
        for block in msg.content:
            if isinstance(block, ToolResultBlock):
                total += len(block.content)
            elif hasattr(block, "text"):
                total += len(block.text)  # type: ignore[union-attr]
            elif hasattr(block, "input"):
                # ToolUseBlock — estimate arguments.
                try:
                    import json
                    total += len(json.dumps(block.input))  # type: ignore[union-attr]
                except Exception:
                    total += 128
    return total
