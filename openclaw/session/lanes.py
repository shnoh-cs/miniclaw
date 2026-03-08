"""Session lanes: parallel conversation threads within a single session.

A lane is an isolated message stream that shares the same session file but
operates independently.  This is useful for:

- Subagent tasks running in parallel
- Background operations (memory flush, compaction) that shouldn't pollute
  the main conversation
- Multi-turn side conversations (e.g. clarification loops)

Each lane has its own message list and can be compacted independently.
The ``LaneManager`` multiplexes lanes over a single ``SessionManager``.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from openclaw.agent.types import AgentMessage, TextBlock


class LaneStatus(str, Enum):
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class Lane:
    """A single conversation lane within a session."""

    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    name: str = ""
    parent_lane_id: str | None = None
    status: LaneStatus = LaneStatus.ACTIVE
    created_at: float = field(default_factory=time.time)
    messages: list[AgentMessage] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def message_count(self) -> int:
        return len(self.messages)

    def append(self, message: AgentMessage) -> None:
        self.messages.append(message)

    def get_text_history(self) -> str:
        """Get a plain-text summary of this lane's conversation."""
        parts: list[str] = []
        for msg in self.messages:
            role = msg.role
            texts = [b.text for b in msg.content if isinstance(b, TextBlock)]
            if texts:
                parts.append(f"[{role}] {' '.join(texts)[:500]}")
        return "\n".join(parts)


class LaneManager:
    """Manages multiple lanes within a session.

    The ``main`` lane is always present and is used for the primary
    conversation.  Additional lanes can be created for parallel work.
    """

    def __init__(self) -> None:
        self._lanes: dict[str, Lane] = {}
        # Create the default main lane
        main = Lane(id="main", name="main")
        self._lanes["main"] = main

    @property
    def main(self) -> Lane:
        return self._lanes["main"]

    def create(
        self,
        name: str = "",
        parent_lane_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Lane:
        """Create a new lane."""
        lane = Lane(
            name=name or f"lane-{len(self._lanes)}",
            parent_lane_id=parent_lane_id,
            metadata=metadata or {},
        )
        self._lanes[lane.id] = lane
        return lane

    def get(self, lane_id: str) -> Lane | None:
        return self._lanes.get(lane_id)

    def list_active(self) -> list[Lane]:
        return [l for l in self._lanes.values() if l.status == LaneStatus.ACTIVE]

    def list_all(self) -> list[Lane]:
        return list(self._lanes.values())

    def complete(self, lane_id: str) -> None:
        lane = self._lanes.get(lane_id)
        if lane:
            lane.status = LaneStatus.COMPLETED

    def fail(self, lane_id: str) -> None:
        lane = self._lanes.get(lane_id)
        if lane:
            lane.status = LaneStatus.FAILED

    def pause(self, lane_id: str) -> None:
        lane = self._lanes.get(lane_id)
        if lane:
            lane.status = LaneStatus.PAUSED

    def resume(self, lane_id: str) -> None:
        lane = self._lanes.get(lane_id)
        if lane and lane.status == LaneStatus.PAUSED:
            lane.status = LaneStatus.ACTIVE

    def remove(self, lane_id: str) -> bool:
        """Remove a lane (cannot remove main)."""
        if lane_id == "main":
            return False
        return self._lanes.pop(lane_id, None) is not None

    def merge_into_main(self, lane_id: str) -> str | None:
        """Merge a lane's conversation summary into the main lane.

        Returns the summary text, or None if lane not found.
        """
        lane = self._lanes.get(lane_id)
        if not lane or lane_id == "main":
            return None

        summary = lane.get_text_history()
        if summary:
            merge_msg = AgentMessage(
                role="user",
                content=[TextBlock(
                    text=f"[Merged from lane '{lane.name}' ({lane.id})]\n{summary}"
                )],
            )
            self.main.append(merge_msg)

        lane.status = LaneStatus.COMPLETED
        return summary
