"""Built-in tool: show session status (time, model, token usage).

The real executor is wired up in Agent.__init__() since it needs
references to the agent instance for model/session info.
"""

from __future__ import annotations

from openclaw.agent.types import ToolDefinition

DEFINITION = ToolDefinition(
    name="session_status",
    description=(
        "Show session status: current date/time, token usage, "
        "thinking level, model. Use when you need the current time."
    ),
    parameters=[],
)
