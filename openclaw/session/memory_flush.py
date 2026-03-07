"""Pre-compaction memory flush: silently save durable memories before context loss."""

from __future__ import annotations

from typing import TYPE_CHECKING

from openclaw.agent.types import AgentMessage, TextBlock

if TYPE_CHECKING:
    from openclaw.config import CompactionConfig
    from openclaw.model.provider import ModelProvider
    from openclaw.session.manager import SessionManager

NO_REPLY = "NO_REPLY"

FLUSH_SYSTEM_PROMPT = (
    "Session nearing compaction. Store durable memories now. "
    "Write any lasting notes, decisions, or important context to memory. "
    "Reply with NO_REPLY if nothing to store."
)

FLUSH_USER_PROMPT = (
    "The session is approaching context compaction. "
    "Write any durable facts, decisions, or important context to "
    "memory/{date}.md before they are lost. "
    "If nothing important to save, reply with NO_REPLY."
)


async def should_flush(
    session: SessionManager,
    config: CompactionConfig,
    context_max_tokens: int,
) -> bool:
    """Check if memory flush should be triggered.

    Triggers when tokens are within soft_threshold of compaction point.
    """
    if not config.memory_flush.enabled:
        return False

    current_tokens = session.estimate_tokens()
    compaction_point = context_max_tokens - config.memory_flush.soft_threshold_tokens
    return current_tokens >= compaction_point


async def execute_memory_flush(
    session: SessionManager,
    provider: ModelProvider,
    workspace_dir: str,
) -> str | None:
    """Execute a silent memory flush turn.

    The agent writes important information to workspace memory files.
    Returns the agent's response (or None if NO_REPLY).
    """
    import datetime

    date_str = datetime.date.today().isoformat()
    user_prompt = FLUSH_USER_PROMPT.replace("{date}", date_str)

    response = await provider.complete(
        system=FLUSH_SYSTEM_PROMPT,
        messages=session.messages + [
            AgentMessage(role="user", content=[TextBlock(text=user_prompt)])
        ],
    )

    if response.strip().upper() == NO_REPLY:
        return None

    return response
