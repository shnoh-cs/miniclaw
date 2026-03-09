"""Tool result truncation and session storage guards."""

from __future__ import annotations

import re

from openclaw.agent.types import ToolResultBlock

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

HARD_LIMIT_CHARS = 400_000
MIN_KEEP_CHARS = 2_000
_TAIL_CAP = 4_000

TRUNCATION_SUFFIX = (
    "\n\n\u26a0\ufe0f [Content truncated \u2014 original was too large for the model's "
    "context window. The content above is a partial view. If you need more, "
    "request specific sections or use offset/limit parameters to read smaller chunks.]"
)

MIDDLE_OMISSION_MARKER = (
    "\n\n\u26a0\ufe0f [... middle content omitted \u2014 showing head and tail ...]\n\n"
)

SESSION_HARD_CAP = 400_000

SESSION_CAP_NOTICE = (
    "\n\n[Tool result capped before session write — original exceeded "
    "session storage limit. Re-run with smaller scope if needed.]"
)

MISSING_TOOL_RESULT_PLACEHOLDER = (
    "[Tool result not received — possibly timed out]"
)

# Patterns that indicate the tail of output is diagnostically important
_IMPORTANT_TAIL_RE = re.compile(
    r"\b(error|exception|failed|fatal|traceback|panic|stack trace|errno|exit code"
    r"|total|summary|result|complete|finished|done)\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _has_important_tail(text: str) -> bool:
    """Detect error/exception/traceback/JSON patterns in the last 2000 chars."""
    tail = text[-2000:]
    if _IMPORTANT_TAIL_RE.search(tail):
        return True
    if re.search(r"\}\s*$", tail.strip()):
        return True
    return False


def _find_newline_cut(text: str, target: int, lo_ratio: float = 0.8) -> int:
    """Find a clean newline cut point within [lo_ratio*target, target]."""
    lo = int(target * lo_ratio)
    nl = text.rfind("\n", lo, target)
    return nl if nl > lo else target


# ---------------------------------------------------------------------------
# Tool result truncation
# ---------------------------------------------------------------------------

def truncate_tool_result(content: str, max_chars: int) -> str:
    """Truncate tool result with smart head+tail strategy.

    Uses head+tail only when the tail contains important diagnostic content.
    Otherwise head-only with a clean newline boundary cut.
    """
    if len(content) <= max_chars:
        return content

    effective_max = max(max_chars, MIN_KEEP_CHARS)
    if effective_max >= len(content):
        return content

    budget = max(MIN_KEEP_CHARS, effective_max - len(TRUNCATION_SUFFIX))

    if _has_important_tail(content) and budget > MIN_KEEP_CHARS * 2:
        tail_budget = min(int(budget * 0.3), _TAIL_CAP)
        head_budget = budget - tail_budget - len(MIDDLE_OMISSION_MARKER)

        if head_budget > MIN_KEEP_CHARS:
            head_cut = _find_newline_cut(content, head_budget)
            tail_start = len(content) - tail_budget
            nl = content.find("\n", tail_start)
            if nl != -1 and nl < tail_start + int(tail_budget * 0.2):
                tail_start = nl + 1
            return (
                content[:head_cut]
                + MIDDLE_OMISSION_MARKER
                + content[tail_start:]
                + TRUNCATION_SUFFIX
            )

    cut_point = _find_newline_cut(content, budget)
    return content[:cut_point] + TRUNCATION_SUFFIX


# ---------------------------------------------------------------------------
# Session storage guard
# ---------------------------------------------------------------------------

def cap_tool_result_for_session(content: str, max_chars: int = SESSION_HARD_CAP) -> str:
    """Hard cap tool result content before writing to session JSONL."""
    if len(content) <= max_chars:
        return content

    budget = max(0, max_chars - len(SESSION_CAP_NOTICE))
    if budget <= 0:
        return SESSION_CAP_NOTICE.lstrip("\n")

    cut = _find_newline_cut(content, budget)
    return content[:cut] + SESSION_CAP_NOTICE


def synthesize_missing_tool_result(tool_use_id: str) -> ToolResultBlock:
    """Create a synthetic result for an orphaned tool_use block."""
    return ToolResultBlock(
        tool_use_id=tool_use_id,
        content=MISSING_TOOL_RESULT_PLACEHOLDER,
        is_error=True,
    )
