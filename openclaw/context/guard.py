"""Context window guard: token budget management with overflow recovery."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto

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
        max_chars = int(self.config.max_tokens * 4 * self.config.tool_result_max_ratio)
        return result_chars <= max_chars

    def tool_result_max_chars(self) -> int:
        """Maximum allowed characters for a single tool result."""
        return int(self.config.max_tokens * 4 * self.config.tool_result_max_ratio)
