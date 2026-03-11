"""Built-in tool: manage cron jobs and wake events.

The real executor is wired up in Agent.__init__() since it needs
references to the scheduler and agent instance.
"""

from __future__ import annotations

from openclaw.agent.types import ToolDefinition, ToolParameter

DEFINITION = ToolDefinition(
    name="cron",
    description=(
        "Manage cron jobs and wake events. Use for reminders and recurring tasks. "
        "Actions: list, create, delete, status. "
        "When scheduling a reminder, write the task text as something that will "
        "read like a reminder when it fires."
    ),
    parameters=[
        ToolParameter(
            name="action",
            description="list | create | delete | status",
        ),
        ToolParameter(
            name="name",
            description="Job name (required for create/delete/status)",
            required=False,
        ),
        ToolParameter(
            name="interval_seconds",
            type="integer",
            description="Interval in seconds (required for create)",
            required=False,
        ),
        ToolParameter(
            name="task",
            description=(
                "Task description — the agent will execute this when the job fires "
                "(required for create)"
            ),
            required=False,
        ),
        ToolParameter(
            name="one_shot",
            type="boolean",
            description="If true, run only once then auto-delete (default: false)",
            required=False,
        ),
    ],
)
