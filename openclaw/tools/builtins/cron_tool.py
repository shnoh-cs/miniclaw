"""Built-in tool: manage cron jobs and wake events.

Supports three schedule types:
- interval_seconds: run every N seconds (e.g., 3600 = every hour)
- cron_expr: cron expression (e.g., "0 7,19 * * *" = 7am and 7pm daily)
- at: one-shot absolute time (e.g., "2026-03-12T07:00:00+09:00")

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
        "Schedule with cron_expr for specific times (e.g. '0 7,19 * * *' for 7am/7pm), "
        "interval_seconds for fixed intervals, or at for one-shot absolute time."
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
            name="cron_expr",
            description=(
                "Cron expression for specific times. "
                "Format: 'minute hour day-of-month month day-of-week'. "
                "Examples: '0 7 * * *' (daily 7am), '0 7,19 * * *' (7am+7pm), "
                "'30 9 * * 1-5' (weekdays 9:30am). "
                "Preferred over interval_seconds when the user specifies exact times."
            ),
            required=False,
        ),
        ToolParameter(
            name="timezone",
            description=(
                "IANA timezone for cron_expr (e.g. 'Asia/Seoul', 'America/New_York'). "
                "Defaults to system timezone if omitted."
            ),
            required=False,
        ),
        ToolParameter(
            name="at",
            description=(
                "ISO 8601 timestamp for one-shot execution "
                "(e.g. '2026-03-12T07:00:00+09:00'). Automatically sets one_shot=true."
            ),
            required=False,
        ),
        ToolParameter(
            name="interval_seconds",
            type="integer",
            description="Fixed interval in seconds. Use cron_expr instead for specific times of day.",
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
