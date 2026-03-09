"""Context window self-diagnosis, recommendations, and auto-adjustment."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from openclaw.agent.types import (
    AgentMessage,
    TextBlock,
    ToolDefinition,
    ToolResultBlock,
    ToolUseBlock,
)

if TYPE_CHECKING:
    from openclaw.config import ContextConfig


@dataclass
class ConfigAdjustment:
    """A single auto-applied configuration change."""

    key: str
    old_value: Any
    new_value: Any
    reason: str


@dataclass
class ContextDiagnosis:
    """Breakdown of context window usage with recommendations."""

    total_tokens: int = 0
    system_prompt_tokens: int = 0
    tool_schemas_tokens: int = 0
    session_history_tokens: int = 0
    compaction_summary_tokens: int = 0
    max_tokens: int = 0
    utilization: float = 0.0
    message_count: int = 0
    tool_result_count: int = 0
    large_result_count: int = 0
    recommendations: list[str] = field(default_factory=list)
    adjustments: list[ConfigAdjustment] = field(default_factory=list)

    def format(self) -> str:
        """Format diagnosis as a human-readable report."""
        bar_len = 30
        filled = int(self.utilization * bar_len)
        bar = "█" * filled + "░" * (bar_len - filled)

        lines = [
            f"Context Usage: [{bar}] {self.utilization:.0%}",
            f"  Total:         ~{self.total_tokens:,} / {self.max_tokens:,} tokens",
            f"  System prompt: ~{self.system_prompt_tokens:,} tokens",
            f"  Tool schemas:  ~{self.tool_schemas_tokens:,} tokens",
            f"  Session:       ~{self.session_history_tokens:,} tokens"
            f" ({self.message_count} messages)",
        ]
        if self.compaction_summary_tokens > 0:
            lines.append(
                f"  Compaction:    ~{self.compaction_summary_tokens:,} tokens"
            )

        if self.large_result_count > 0:
            lines.append(f"  Large results: {self.large_result_count} (>10K chars)")

        if self.adjustments:
            lines.append("")
            lines.append("  Auto-adjustments applied:")
            for adj in self.adjustments:
                lines.append(
                    f"    {adj.key}: {adj.old_value} → {adj.new_value} ({adj.reason})"
                )

        if self.recommendations:
            lines.append("")
            for rec in self.recommendations:
                lines.append(f"  → {rec}")

        return "\n".join(lines)

    def apply_adjustments(self, context_config: ContextConfig) -> list[ConfigAdjustment]:
        """Auto-apply configuration adjustments based on diagnosis.

        Mutates context_config in-place. Returns the list of adjustments made.
        """
        adjustments: list[ConfigAdjustment] = []

        if self.utilization >= 0.85:
            # Increase reserve floor to force earlier compaction
            old_floor = context_config.reserve_tokens_floor
            new_floor = min(old_floor + 5000, self.max_tokens // 2)
            if new_floor > old_floor:
                adj = ConfigAdjustment(
                    key="reserve_tokens_floor",
                    old_value=old_floor,
                    new_value=new_floor,
                    reason="context at 85%+, force earlier compaction",
                )
                context_config.reserve_tokens_floor = new_floor
                adjustments.append(adj)

            # Reduce tool result ratio
            old_ratio = context_config.tool_result_max_ratio
            new_ratio = max(0.15, round(old_ratio - 0.05, 2))
            if new_ratio < old_ratio:
                adj = ConfigAdjustment(
                    key="tool_result_max_ratio",
                    old_value=old_ratio,
                    new_value=new_ratio,
                    reason="context at 85%+, reduce tool output budget",
                )
                context_config.tool_result_max_ratio = new_ratio
                adjustments.append(adj)

        elif self.utilization >= 0.70:
            # Lower compaction threshold to trigger earlier
            old_threshold = context_config.compaction_threshold
            new_threshold = max(0.5, round(old_threshold - 0.05, 2))
            if new_threshold < old_threshold:
                adj = ConfigAdjustment(
                    key="compaction_threshold",
                    old_value=old_threshold,
                    new_value=new_threshold,
                    reason="context at 70%+, trigger compaction sooner",
                )
                context_config.compaction_threshold = new_threshold
                adjustments.append(adj)

        self.adjustments = adjustments
        return adjustments


def diagnose_context(
    messages: list[AgentMessage],
    system_prompt: str,
    tool_definitions: list[ToolDefinition] | None = None,
    max_tokens: int = 32768,
    compaction_summary: str | None = None,
) -> ContextDiagnosis:
    """Analyze context window usage and provide actionable recommendations."""

    # Estimate tokens (rough: 4 chars per token)
    system_tokens = len(system_prompt) // 4

    tool_tokens = 0
    if tool_definitions:
        for td in tool_definitions:
            tool_tokens += len(td.name) + len(td.description)
            for p in td.parameters:
                tool_tokens += len(p.name) + len(p.description)
        tool_tokens = tool_tokens // 4

    compaction_tokens = len(compaction_summary) // 4 if compaction_summary else 0

    session_tokens = 0
    tool_result_count = 0
    large_result_count = 0
    for m in messages:
        for block in m.content:
            if isinstance(block, ToolResultBlock):
                tool_result_count += 1
                session_tokens += len(block.content) // 4
                if len(block.content) > 10000:
                    large_result_count += 1
            elif isinstance(block, TextBlock):
                session_tokens += len(block.text) // 4
            elif isinstance(block, ToolUseBlock):
                try:
                    session_tokens += len(json.dumps(block.input)) // 4
                except Exception:
                    session_tokens += 32

    total = system_tokens + tool_tokens + session_tokens + compaction_tokens
    utilization = total / max_tokens if max_tokens > 0 else 1.0

    # Build recommendations
    recs: list[str] = []

    if utilization >= 0.85:
        recs.append("CRITICAL: Context at 85%+. Compaction imminent.")
        recs.append("Use subagents for remaining subtasks to preserve main context.")
    elif utilization >= 0.70:
        recs.append("Context at 70%+. Compaction will trigger soon.")
        recs.append("Consider delegating complex subtasks to subagents.")
    elif utilization >= 0.50:
        recs.append("Context at 50%. Monitor growth rate.")

    if large_result_count >= 3:
        recs.append(
            f"{large_result_count} large tool results detected. "
            f"Consider reducing output verbosity."
        )

    if session_tokens > system_tokens * 5 and system_tokens > 0:
        recs.append(
            "Session history dominates context. "
            "Multi-turn conversations drain budget fast."
        )

    if max_tokens > 0 and tool_tokens > max_tokens * 0.1:
        recs.append(
            f"Tool schemas use ~{tool_tokens:,} tokens "
            f"({tool_tokens * 100 // max_tokens}%). "
            f"Consider reducing tool count."
        )

    return ContextDiagnosis(
        total_tokens=total,
        system_prompt_tokens=system_tokens,
        tool_schemas_tokens=tool_tokens,
        session_history_tokens=session_tokens,
        compaction_summary_tokens=compaction_tokens,
        max_tokens=max_tokens,
        utilization=utilization,
        message_count=len(messages),
        tool_result_count=tool_result_count,
        large_result_count=large_result_count,
        recommendations=recs,
    )
