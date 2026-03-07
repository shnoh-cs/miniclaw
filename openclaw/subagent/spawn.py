"""Subagent system: spawning, lifecycle, depth control, and announce flow."""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable

from openclaw.agent.types import AgentMessage, RunResult, TextBlock, ToolDefinition

# Tools excluded from subagents by default
SUBAGENT_EXCLUDED_TOOLS = {
    "sessions_list", "sessions_history", "sessions_send", "sessions_spawn",
}

# Tools given back to orchestrators (depth 1 with maxSpawnDepth >= 2)
ORCHESTRATOR_TOOLS = {
    "sessions_spawn", "sessions_list", "sessions_history",
}

DEFAULT_MAX_SPAWN_DEPTH = 1
DEFAULT_MAX_CHILDREN = 5
DEFAULT_TIMEOUT_SECONDS = 600


class SubagentMode(str, Enum):
    RUN = "run"        # one-shot execution
    SESSION = "session"  # persistent thread-bound


class SubagentStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMED_OUT = "timed_out"


@dataclass
class SubagentConfig:
    max_spawn_depth: int = DEFAULT_MAX_SPAWN_DEPTH
    max_children_per_agent: int = DEFAULT_MAX_CHILDREN
    default_timeout: int = DEFAULT_TIMEOUT_SECONDS
    default_model: str = ""  # empty = inherit from parent
    default_thinking: str = ""  # empty = inherit


@dataclass
class SubagentResult:
    """Result announced back to the parent."""

    subagent_id: str
    status: SubagentStatus
    text: str = ""
    error: str | None = None
    duration_seconds: float = 0.0
    usage: dict[str, int] = field(default_factory=dict)


@dataclass
class SubagentEntry:
    """Registry entry for a spawned subagent."""

    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    parent_session_key: str = ""
    session_key: str = ""
    task: str = ""
    mode: SubagentMode = SubagentMode.RUN
    model: str = ""
    thinking: str = ""
    depth: int = 0
    status: SubagentStatus = SubagentStatus.PENDING
    created_at: float = field(default_factory=time.time)
    completed_at: float | None = None
    result: SubagentResult | None = None
    children: list[str] = field(default_factory=list)


class SubagentRegistry:
    """Manages subagent lifecycle: spawn → run → complete → announce."""

    def __init__(self, config: SubagentConfig | None = None) -> None:
        self.config = config or SubagentConfig()
        self._entries: dict[str, SubagentEntry] = {}
        self._on_announce: Callable[[SubagentResult], None] | None = None

    def set_announce_handler(self, handler: Callable[[SubagentResult], None]) -> None:
        self._on_announce = handler

    def can_spawn(self, parent_depth: int, parent_session_key: str) -> tuple[bool, str]:
        """Check if a new subagent can be spawned."""
        if parent_depth >= self.config.max_spawn_depth:
            return False, f"Max spawn depth ({self.config.max_spawn_depth}) reached"

        # Count active children of this parent
        active = sum(
            1 for e in self._entries.values()
            if e.parent_session_key == parent_session_key
            and e.status in {SubagentStatus.PENDING, SubagentStatus.RUNNING}
        )
        if active >= self.config.max_children_per_agent:
            return False, f"Max children ({self.config.max_children_per_agent}) reached"

        return True, ""

    def spawn(
        self,
        *,
        parent_session_key: str,
        task: str,
        depth: int,
        mode: SubagentMode = SubagentMode.RUN,
        model: str = "",
        thinking: str = "",
    ) -> SubagentEntry:
        """Create a new subagent entry."""
        entry = SubagentEntry(
            parent_session_key=parent_session_key,
            session_key=f"{parent_session_key}:subagent:{uuid.uuid4().hex[:8]}",
            task=task,
            mode=mode,
            model=model or self.config.default_model,
            thinking=thinking or self.config.default_thinking,
            depth=depth + 1,
        )
        self._entries[entry.id] = entry
        return entry

    def mark_running(self, subagent_id: str) -> None:
        entry = self._entries.get(subagent_id)
        if entry:
            entry.status = SubagentStatus.RUNNING

    def mark_completed(
        self, subagent_id: str, text: str = "", error: str | None = None
    ) -> None:
        """Mark subagent as completed and trigger announce."""
        entry = self._entries.get(subagent_id)
        if not entry:
            return

        now = time.time()
        entry.completed_at = now
        entry.status = SubagentStatus.FAILED if error else SubagentStatus.COMPLETED

        result = SubagentResult(
            subagent_id=subagent_id,
            status=entry.status,
            text=text,
            error=error,
            duration_seconds=now - entry.created_at,
        )
        entry.result = result

        # Announce to parent
        if self._on_announce:
            self._on_announce(result)

    def mark_timed_out(self, subagent_id: str) -> None:
        entry = self._entries.get(subagent_id)
        if entry:
            entry.status = SubagentStatus.TIMED_OUT
            entry.completed_at = time.time()

    def get_active(self) -> list[SubagentEntry]:
        return [
            e for e in self._entries.values()
            if e.status in {SubagentStatus.PENDING, SubagentStatus.RUNNING}
        ]

    def cascade_stop(self, session_key: str) -> int:
        """Stop all subagents under a session key. Returns count stopped."""
        stopped = 0
        for entry in self._entries.values():
            if (
                entry.parent_session_key == session_key
                and entry.status in {SubagentStatus.PENDING, SubagentStatus.RUNNING}
            ):
                entry.status = SubagentStatus.FAILED
                entry.completed_at = time.time()
                stopped += 1
                # Cascade to children
                stopped += self.cascade_stop(entry.session_key)
        return stopped

    def get_tools_for_depth(
        self, depth: int, all_tools: list[ToolDefinition]
    ) -> list[ToolDefinition]:
        """Filter tools based on subagent depth.

        - Depth 0 (main): all tools
        - Depth 1 (leaf when maxDepth=1): exclude session tools
        - Depth 1 (orchestrator when maxDepth>=2): include session tools
        - Depth 2+: exclude sessions_spawn (no further nesting)
        """
        if depth == 0:
            return all_tools

        is_orchestrator = depth == 1 and self.config.max_spawn_depth >= 2

        filtered = []
        for tool in all_tools:
            if is_orchestrator:
                # Orchestrator gets session tools
                filtered.append(tool)
            elif tool.name in SUBAGENT_EXCLUDED_TOOLS:
                continue
            elif depth >= 2 and tool.name == "sessions_spawn":
                continue
            else:
                filtered.append(tool)

        return filtered
