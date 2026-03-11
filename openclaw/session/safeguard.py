"""Safeguard validation, tool failure tracking, and file operations for compaction."""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

from openclaw.agent.types import (
    AgentMessage,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
)
from openclaw.session.identifiers import _summary_includes_identifier

if TYPE_CHECKING:
    from openclaw.model.provider import ModelProvider

log = logging.getLogger("openclaw.compaction.safeguard")

# Limits
MAX_TOOL_FAILURES = 8
MAX_TOOL_FAILURE_CHARS = 240
MAX_ASK_OVERLAP_TOKENS = 12
MIN_ASK_OVERLAP_TOKENS_FOR_DOUBLE_MATCH = 3

# Required sections in a safeguard-validated summary (order matters)
_REQUIRED_SECTIONS = [
    "## Decisions",
    "## Completed work",
    "## Open TODOs",
    "## Constraints",
    "## Pending user asks",
    "## Exact identifiers",
]

# Basic English stop words for ask-overlap filtering
_STOP_WORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "must",
    "i", "me", "my", "we", "our", "you", "your", "he", "she", "it",
    "they", "them", "his", "her", "its", "this", "that", "these", "those",
    "in", "on", "at", "to", "for", "of", "with", "by", "from", "up",
    "about", "into", "through", "during", "before", "after", "above",
    "below", "between", "and", "but", "or", "nor", "not", "no", "so",
    "if", "then", "than", "too", "very", "just", "also", "only",
    "what", "which", "who", "whom", "how", "when", "where", "why",
    "all", "each", "every", "both", "few", "more", "most", "other",
    "some", "such", "here", "there", "again", "once",
})


# ---------------------------------------------------------------------------
# Tool failure tracking
# ---------------------------------------------------------------------------

class ToolFailure:
    """Represents a single tool call failure."""

    def __init__(self, tool_use_id: str, tool_name: str, summary: str, meta: str | None = None):
        self.tool_use_id = tool_use_id
        self.tool_name = tool_name
        self.summary = summary
        self.meta = meta


def _normalize_failure_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _truncate_failure_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 3)] + "..."


def collect_tool_failures(messages: list[AgentMessage]) -> list[ToolFailure]:
    """Collect error tool results from messages, deduped by tool_use_id."""
    failures: list[ToolFailure] = []
    seen: set[str] = set()

    for msg in messages:
        for block in msg.content:
            if not isinstance(block, ToolResultBlock):
                continue
            if not block.is_error:
                continue
            tid = block.tool_use_id
            if not tid or tid in seen:
                continue
            seen.add(tid)

            raw = _normalize_failure_text(block.content if isinstance(block.content, str) else "")
            summary = _truncate_failure_text(raw or "failed (no output)", MAX_TOOL_FAILURE_CHARS)
            failures.append(ToolFailure(tool_use_id=tid, tool_name="tool", summary=summary))

    return failures


def format_tool_failures_section(failures: list[ToolFailure]) -> str:
    if not failures:
        return ""
    lines = []
    for f in failures[:MAX_TOOL_FAILURES]:
        meta = f" ({f.meta})" if f.meta else ""
        lines.append(f"- {f.tool_name}{meta}: {f.summary}")
    if len(failures) > MAX_TOOL_FAILURES:
        lines.append(f"- ...and {len(failures) - MAX_TOOL_FAILURES} more")
    return "\n\n## Tool Failures\n" + "\n".join(lines)


# ---------------------------------------------------------------------------
# File operations tracking
# ---------------------------------------------------------------------------

def collect_file_operations(messages: list[AgentMessage]) -> tuple[list[str], list[str]]:
    """Collect read and modified file paths from tool calls in messages.

    Returns (read_files, modified_files) sorted lists.
    """
    read_files: set[str] = set()
    modified_files: set[str] = set()

    read_tools = {"Read", "read", "cat", "head", "tail"}
    write_tools = {"Write", "write", "Edit", "edit", "ApplyPatch", "apply_patch", "Bash", "bash"}

    for msg in messages:
        for block in msg.content:
            if not isinstance(block, ToolUseBlock):
                continue
            tool_input = block.input if isinstance(block.input, dict) else {}
            tool_name = block.name

            file_path = (
                tool_input.get("file_path")
                or tool_input.get("path")
                or tool_input.get("filename")
            )

            if not file_path or not isinstance(file_path, str):
                continue

            if tool_name in write_tools:
                modified_files.add(file_path)
            elif tool_name in read_tools:
                read_files.add(file_path)

    read_only = read_files - modified_files
    return sorted(read_only), sorted(modified_files)


def format_file_operations(read_files: list[str], modified_files: list[str]) -> str:
    """Format file operations as <read-files> and <modified-files> sections."""
    sections: list[str] = []
    if read_files:
        sections.append(f"<read-files>\n" + "\n".join(read_files) + "\n</read-files>")
    if modified_files:
        sections.append(f"<modified-files>\n" + "\n".join(modified_files) + "\n</modified-files>")
    if not sections:
        return ""
    return "\n\n" + "\n\n".join(sections)


# ---------------------------------------------------------------------------
# Ask overlap validation
# ---------------------------------------------------------------------------

def _tokenize_for_overlap(text: str) -> list[str]:
    """Tokenize text for overlap checking, filtering stop words."""
    normalized = text.lower().strip()
    if not normalized:
        return []
    tokens = re.split(r"[^\w]+", normalized)
    tokens = [t for t in tokens if t and len(t) > 1]
    meaningful = [t for t in tokens if t not in _STOP_WORDS]
    return meaningful if meaningful else tokens


def _has_ask_overlap(summary: str, latest_ask: str | None) -> bool:
    """Check that the summary reflects the latest user ask by keyword overlap."""
    if not latest_ask:
        return True
    ask_tokens = list(dict.fromkeys(_tokenize_for_overlap(latest_ask)))[:MAX_ASK_OVERLAP_TOKENS]
    if not ask_tokens:
        return True
    summary_tokens = set(_tokenize_for_overlap(summary))
    overlap = sum(1 for t in ask_tokens if t in summary_tokens)
    required = 2 if len(ask_tokens) >= MIN_ASK_OVERLAP_TOKENS_FOR_DOUBLE_MATCH else 1
    return overlap >= required


def extract_latest_user_ask(messages: list[AgentMessage]) -> str | None:
    """Find the latest user message text."""
    for msg in reversed(messages):
        if msg.role == "user" and msg.text:
            return msg.text
    return None


# ---------------------------------------------------------------------------
# Ordered section validation
# ---------------------------------------------------------------------------

def _has_required_sections_in_order(summary: str) -> bool:
    """Validate that required sections appear in correct sequential order."""
    lines = [line.strip() for line in summary.split("\n") if line.strip()]
    cursor = 0
    for heading in _REQUIRED_SECTIONS:
        found = False
        for i in range(cursor, len(lines)):
            if lines[i] == heading:
                cursor = i + 1
                found = True
                break
        if not found:
            return False
    return True


# ---------------------------------------------------------------------------
# Safeguard audit
# ---------------------------------------------------------------------------

def audit_summary_quality(
    summary: str,
    identifiers: list[str],
    latest_ask: str | None,
    identifier_policy: str = "strict",
) -> tuple[bool, list[str]]:
    """Audit summary quality. Returns (ok, list_of_reasons)."""
    reasons: list[str] = []

    if not _has_required_sections_in_order(summary):
        lines = [line.strip() for line in summary.split("\n") if line.strip()]
        for section in _REQUIRED_SECTIONS:
            if section not in lines:
                reasons.append(f"missing_section:{section}")
        if not reasons:
            reasons.append("sections_out_of_order")

    if identifier_policy == "strict" and identifiers:
        missing = [
            ident for ident in identifiers
            if not _summary_includes_identifier(summary, ident)
        ]
        if missing:
            reasons.append(f"missing_identifiers:{','.join(missing[:3])}")

    if not _has_ask_overlap(summary, latest_ask):
        reasons.append("latest_user_ask_not_reflected")

    return (len(reasons) == 0, reasons)


# ---------------------------------------------------------------------------
# Safeguard validation with retry
# ---------------------------------------------------------------------------

async def safeguard_validate(
    summary: str,
    identifiers: list[str],
    provider: ModelProvider,
    max_retries: int,
    original_messages: list[AgentMessage],
    latest_ask: str | None,
    identifier_policy: str,
    last_successful_summary: str | None,
) -> str:
    """Validate summary quality and retry by re-summarizing from original messages."""
    from openclaw.session.compaction import (
        _build_summarization_prompt,
        _llm_complete_with_retry,
        _messages_to_text,
    )

    capped_retries = min(max_retries, 3)

    for attempt in range(capped_retries + 1):
        ok, reasons = audit_summary_quality(
            summary, identifiers, latest_ask, identifier_policy
        )

        if ok:
            return summary

        if attempt >= capped_retries:
            log.warning(
                "Safeguard validation exhausted %d retries, reasons: %s",
                capped_retries, ", ".join(reasons),
            )
            break

        log.info(
            "Safeguard retry %d/%d, re-summarizing. Reasons: %s",
            attempt + 1, capped_retries, ", ".join(reasons),
        )

        quality_feedback = (
            f"Previous summary failed quality checks ({', '.join(reasons)}). "
            "Fix all issues and include every required section with exact identifiers preserved."
        )

        try:
            context_text = _messages_to_text(original_messages)
            prompt = _build_summarization_prompt(
                context_text, identifier_policy, identifiers, None
            )
            prompt += f"\n\nQuality feedback from previous attempt:\n{quality_feedback}"

            summary = await _llm_complete_with_retry(
                provider,
                system="You are a precise summarizer. Follow the structure exactly.",
                messages=[AgentMessage(role="user", content=[TextBlock(text=prompt)])],
            )
        except Exception as e:
            log.warning(
                "Safeguard re-summarization failed on attempt %d: %s",
                attempt + 1, e,
            )
            if last_successful_summary:
                log.info("Falling back to last successful summary.")
                return last_successful_summary
            break

    return summary
