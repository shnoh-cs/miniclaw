"""Pre-compaction memory flush: silently save durable memories before context loss."""

from __future__ import annotations

from typing import TYPE_CHECKING

from openclaw.agent.types import AgentMessage, TextBlock

if TYPE_CHECKING:
    from openclaw.config import CompactionConfig, WorkspaceConfig
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

# Tracks the compaction count at which the last flush was executed.
# This prevents double-flushing within the same compaction cycle:
# once a flush runs, it won't run again until a compaction actually occurs
# and increments the count.
_last_flush_compaction_count: dict[str, int] = {}


async def should_flush(
    session: SessionManager,
    config: CompactionConfig,
    context_max_tokens: int,
    workspace_access: str = "rw",
) -> bool:
    """Check if memory flush should be triggered.

    Guards:
    - workspace_access must be "rw" (skip for "ro" or "none")
    - Only flush once per compaction cycle (tracked by compaction count)

    Two-tier token check:
    1. Soft threshold: tokens within soft_threshold of compaction point
    2. Safety margin: tokens >= 85% of context window (catches rapid growth)
    """
    if not config.memory_flush.enabled:
        return False

    # Skip flush in read-only or sandboxed workspaces
    if workspace_access != "rw":
        return False

    current_tokens = session.estimate_tokens()

    # Primary: soft threshold near compaction point
    compaction_point = context_max_tokens - config.memory_flush.soft_threshold_tokens
    in_threshold = current_tokens >= compaction_point

    # Safety margin: 85% of context window (catch rapid token growth)
    safety_threshold = int(context_max_tokens * 0.85)
    in_safety = current_tokens >= safety_threshold

    if not (in_threshold or in_safety):
        return False

    # Double-flush guard: only flush once per compaction cycle
    session_key = session.session_id
    current_compaction_count = len(session.compaction_entries)
    last_flushed = _last_flush_compaction_count.get(session_key, -1)
    if last_flushed >= current_compaction_count:
        return False  # already flushed in this cycle

    return True


def mark_flushed(session: SessionManager) -> None:
    """Mark that a flush was executed for the current compaction cycle."""
    _last_flush_compaction_count[session.session_id] = len(session.compaction_entries)


async def execute_memory_flush(
    session: SessionManager,
    provider: ModelProvider,
    workspace_dir: str,
) -> str | None:
    """Execute a silent memory flush turn.

    The agent writes important information to workspace memory files.
    The model's response is directly written to a dated memory file.
    Returns the agent's response (or None if NO_REPLY).
    """
    import datetime
    from pathlib import Path

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

    # Write the model's response directly to a dated memory file
    memory_dir = Path(workspace_dir) / "memory"
    memory_dir.mkdir(parents=True, exist_ok=True)
    memory_file = memory_dir / f"{date_str}.md"

    try:
        with open(memory_file, "a", encoding="utf-8") as f:
            f.write(f"\n## Memory Flush ({datetime.datetime.now().isoformat()})\n")
            f.write(response.strip())
            f.write("\n")
    except OSError:
        pass  # flush failure should not crash the agent

    return response
