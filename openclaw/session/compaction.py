"""Multi-stage compaction with identifier preservation and safeguard validation."""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import TYPE_CHECKING

from openclaw.agent.types import (
    AgentMessage,
    CompactionEntry,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
)

if TYPE_CHECKING:
    from openclaw.config import CompactionConfig
    from openclaw.model.provider import ModelProvider
    from openclaw.session.manager import SessionManager

log = logging.getLogger("openclaw.compaction")

# Chunk ratios for multi-stage summarization
BASE_CHUNK_RATIO = 0.4
MIN_CHUNK_RATIO = 0.15
OVERHEAD_TOKENS = 4096

# Limits aligned with the original safeguard implementation
MAX_EXTRACTED_IDENTIFIERS = 12
MAX_TOOL_FAILURES = 8
MAX_TOOL_FAILURE_CHARS = 240
MAX_ASK_OVERLAP_TOKENS = 12
MIN_ASK_OVERLAP_TOKENS_FOR_DOUBLE_MATCH = 3
MAX_LLM_RETRIES = 3
LLM_RETRY_BASE_DELAY = 1.0  # seconds, doubles each retry

# Identifier patterns to preserve during compaction
_IDENTIFIER_PATTERNS = [
    re.compile(r"[0-9a-fA-F]{8,}"),  # hex strings
    re.compile(r"https?://\S+"),  # URLs
    re.compile(r"\b[\w.-]+\.(ts|py|js|go|rs|md|json|yaml|toml)\b"),  # file paths
    re.compile(r"/[\w.-]{2,}(?:/[\w.-]+)+"),  # absolute paths
    re.compile(r"\b\d{1,3}(\.\d{1,3}){3}\b"),  # IPs
    re.compile(r":\d{2,5}\b"),  # ports
    re.compile(r"[A-Za-z0-9_-]{20,}"),  # long tokens/API keys
    re.compile(r"\b\d{6,}\b"),  # long numeric IDs
]

# Required sections in a safeguard-validated summary (order matters)
_REQUIRED_SECTIONS = [
    "## Decisions",
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
# Identifier helpers (sanitize, normalize, extract)
# ---------------------------------------------------------------------------

def _sanitize_identifier(value: str) -> str:
    """Strip surrounding punctuation from an extracted identifier."""
    return (
        value.strip()
        .lstrip("(\"'`[{<")
        .rstrip(")\"'`,;:.!?>]}")
    )


def _is_pure_hex(value: str) -> bool:
    return bool(re.fullmatch(r"[A-Fa-f0-9]{8,}", value))


def _normalize_identifier(value: str) -> str:
    """Normalize an identifier for comparison (hex -> uppercase)."""
    if _is_pure_hex(value):
        return value.upper()
    return value


def _summary_includes_identifier(summary: str, identifier: str) -> bool:
    """Check if a summary contains an identifier (case-insensitive for hex)."""
    if _is_pure_hex(identifier):
        return identifier.upper() in summary.upper()
    return identifier in summary


def extract_identifiers(text: str) -> list[str]:
    """Extract identifiers from text, sanitized and normalized, capped at MAX_EXTRACTED_IDENTIFIERS."""
    seen: set[str] = set()
    result: list[str] = []
    for pattern in _IDENTIFIER_PATTERNS:
        for match in pattern.finditer(text):
            sanitized = _sanitize_identifier(match.group(0))
            normalized = _normalize_identifier(sanitized)
            if len(normalized) >= 4 and normalized not in seen:
                seen.add(normalized)
                result.append(normalized)
    return result[:MAX_EXTRACTED_IDENTIFIERS]


def _extract_identifiers_from_recent(messages: list[AgentMessage], max_messages: int = 10) -> list[str]:
    """Extract identifiers from the last N messages only (capped at MAX_EXTRACTED_IDENTIFIERS)."""
    recent = messages[-max_messages:] if len(messages) > max_messages else messages
    text = _messages_to_text(recent)
    return extract_identifiers(text)


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


def _format_tool_failures_section(failures: list[ToolFailure]) -> str:
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

def _collect_file_operations(messages: list[AgentMessage]) -> tuple[list[str], list[str]]:
    """Collect read and modified file paths from tool calls in messages.

    Returns (read_files, modified_files) sorted lists.
    """
    read_files: set[str] = set()
    modified_files: set[str] = set()

    # Tool names that indicate file reads vs modifications
    read_tools = {"Read", "read", "cat", "head", "tail"}
    write_tools = {"Write", "write", "Edit", "edit", "ApplyPatch", "apply_patch", "Bash", "bash"}

    for msg in messages:
        for block in msg.content:
            if not isinstance(block, ToolUseBlock):
                continue
            tool_input = block.input if isinstance(block.input, dict) else {}
            tool_name = block.name

            # Extract file path from common parameter names
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

    # Files that were modified should not appear in read-only list
    read_only = read_files - modified_files
    return sorted(read_only), sorted(modified_files)


def _format_file_operations(read_files: list[str], modified_files: list[str]) -> str:
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
    # Split on non-alphanumeric (Unicode-aware)
    tokens = re.split(r"[^\w]+", normalized)
    tokens = [t for t in tokens if t and len(t) > 1]
    # Filter stop words
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


def _extract_latest_user_ask(messages: list[AgentMessage]) -> str | None:
    """Find the latest user message text."""
    for msg in reversed(messages):
        if msg.role == "user" and msg.text:
            return msg.text
    return None


# ---------------------------------------------------------------------------
# Ordered section validation
# ---------------------------------------------------------------------------

def _has_required_sections_in_order(summary: str) -> bool:
    """Validate that required sections appear in correct sequential order (not just substring presence)."""
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
# Core helpers
# ---------------------------------------------------------------------------

def strip_tool_result_details(messages: list[AgentMessage]) -> list[AgentMessage]:
    """Strip verbose tool result content for security (before sending to summarizer)."""
    stripped: list[AgentMessage] = []
    for msg in messages:
        new_blocks = []
        for block in msg.content:
            if isinstance(block, ToolResultBlock) and len(block.content) > 2000:
                new_blocks.append(
                    ToolResultBlock(
                        tool_use_id=block.tool_use_id,
                        content=block.content[:1000] + "\n...[truncated]...\n" + block.content[-500:],
                        is_error=block.is_error,
                    )
                )
            else:
                new_blocks.append(block)
        stripped.append(msg.model_copy(update={"content": new_blocks}))
    return stripped


def _estimate_tokens(text: str) -> int:
    return len(text) // 4


def _messages_to_text(messages: list[AgentMessage]) -> str:
    """Convert messages to a text representation for summarization."""
    parts: list[str] = []
    for msg in messages:
        role = msg.role.upper()
        text = msg.text
        tool_uses = msg.tool_uses
        tool_results = msg.tool_results

        if text:
            parts.append(f"[{role}]: {text}")
        for tu in tool_uses:
            parts.append(f"[TOOL_USE: {tu.name}]: {json.dumps(tu.input)[:500]}")
        for tr in tool_results:
            status = "ERROR" if tr.is_error else "OK"
            parts.append(f"[TOOL_RESULT ({status})]: {tr.content[:500]}")
    return "\n".join(parts)


def _build_summarization_prompt(
    context_text: str,
    identifier_policy: str,
    identifiers: list[str],
    previous_summary: str | None = None,
) -> str:
    """Build the summarization instruction prompt."""
    prompt = (
        "Summarize the following conversation into a compact summary. "
        "Preserve the most important information for continuing the work.\n\n"
        "Produce a compact, factual summary with these exact section headings:\n"
        "## Decisions\nKey decisions made.\n"
        "## Open TODOs\nTasks that are still pending.\n"
        "## Constraints\nRules or constraints discovered.\n"
        "## Pending user asks\nRequests from the user not yet addressed.\n"
        "## Exact identifiers\nCritical IDs, hashes, URLs, file paths.\n"
        "Do not omit unresolved asks from the user.\n\n"
    )

    if identifier_policy == "strict" and identifiers:
        id_list = "\n".join(f"- {id_}" for id_ in identifiers)
        prompt += (
            "For ## Exact identifiers, preserve literal values exactly as seen "
            "(IDs, URLs, file paths, ports, hashes, dates, times).\n"
            f"IMPORTANT: You MUST preserve these exact identifiers:\n{id_list}\n\n"
        )
    elif identifier_policy == "off":
        prompt += (
            "For ## Exact identifiers, include identifiers only when needed for "
            "continuity; do not enforce literal-preservation rules.\n\n"
        )

    if previous_summary:
        prompt += f"Previous summary to incorporate:\n{previous_summary}\n\n"

    prompt += f"Conversation to summarize:\n{context_text}"
    return prompt


# ---------------------------------------------------------------------------
# LLM call with retry + exponential backoff
# ---------------------------------------------------------------------------

async def _llm_complete_with_retry(
    provider: ModelProvider,
    system: str,
    messages: list[AgentMessage],
    max_retries: int = MAX_LLM_RETRIES,
) -> str:
    """Call LLM with simple retry logic and exponential backoff on transient failures."""
    last_error: Exception | None = None
    for attempt in range(max_retries):
        try:
            result = await provider.complete(
                system=system,
                messages=messages,
                model=None,
            )
            return result
        except Exception as e:
            last_error = e
            # Check for non-transient errors (auth, billing) - don't retry those
            err_str = str(e).lower()
            if any(keyword in err_str for keyword in ("auth", "billing", "invalid_api_key", "permission")):
                raise
            if attempt < max_retries - 1:
                delay = LLM_RETRY_BASE_DELAY * (2 ** attempt)
                log.warning(
                    "LLM call failed (attempt %d/%d), retrying in %.1fs: %s",
                    attempt + 1, max_retries, delay, e,
                )
                await asyncio.sleep(delay)
    raise last_error  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Safeguard audit
# ---------------------------------------------------------------------------

def _audit_summary_quality(
    summary: str,
    identifiers: list[str],
    latest_ask: str | None,
    identifier_policy: str = "strict",
) -> tuple[bool, list[str]]:
    """Audit summary quality. Returns (ok, list_of_reasons)."""
    reasons: list[str] = []

    # Ordered section validation (sequential scan)
    if not _has_required_sections_in_order(summary):
        # Find which sections are missing to give detailed feedback
        lines = [line.strip() for line in summary.split("\n") if line.strip()]
        for section in _REQUIRED_SECTIONS:
            if section not in lines:
                reasons.append(f"missing_section:{section}")
        if not reasons:
            # Sections exist but in wrong order
            reasons.append("sections_out_of_order")

    # Strict identifier validation: flag if ANY identifier is missing
    if identifier_policy == "strict" and identifiers:
        missing = [
            ident for ident in identifiers
            if not _summary_includes_identifier(summary, ident)
        ]
        if missing:
            reasons.append(f"missing_identifiers:{','.join(missing[:3])}")

    # Ask overlap validation
    if not _has_ask_overlap(summary, latest_ask):
        reasons.append("latest_user_ask_not_reflected")

    return (len(reasons) == 0, reasons)


# ---------------------------------------------------------------------------
# Main compaction flow
# ---------------------------------------------------------------------------

async def compact_session(
    session: SessionManager,
    provider: ModelProvider,
    config: CompactionConfig,
    context_max_tokens: int,
) -> CompactionEntry | None:
    """Perform multi-stage compaction on the session.

    1. Estimate tokens of all messages
    2. Determine how many old messages to summarize
    3. Generate summary (multi-stage if needed)
    4. Validate with safeguard (if enabled)
    5. Replace old messages with summary
    """
    messages = session.messages
    if len(messages) < 4:
        return None  # not enough to compact

    tokens_before = session.estimate_tokens()
    target_tokens = int(context_max_tokens * 0.5)  # aim for 50% utilization

    # Find split point: keep recent messages, summarize old ones
    keep_count = 4  # minimum messages to keep (2 turns)
    summarize_msgs = messages[:-keep_count] if len(messages) > keep_count else []

    if not summarize_msgs:
        return None

    # Strip verbose tool results before summarization
    stripped = strip_tool_result_details(summarize_msgs)

    # Extract identifiers from last 10 messages only, capped at 12
    identifiers = (
        _extract_identifiers_from_recent(stripped, max_messages=10)
        if config.identifier_policy != "off"
        else []
    )

    # Collect file operations from all messages being summarized
    read_files, modified_files = _collect_file_operations(summarize_msgs)

    # Collect tool failures
    tool_failures = collect_tool_failures(summarize_msgs)

    # Multi-stage chunking if context is too large
    chunks = _chunk_messages(stripped, context_max_tokens)
    previous_summary = session.latest_compaction_summary

    summary = ""
    for chunk in chunks:
        chunk_text = _messages_to_text(chunk)
        prompt = _build_summarization_prompt(
            chunk_text, config.identifier_policy, identifiers, previous_summary
        )

        summary = await _llm_complete_with_retry(
            provider,
            system="You are a precise summarizer. Follow the structure exactly.",
            messages=[AgentMessage(role="user", content=[TextBlock(text=prompt)])],
        )
        previous_summary = summary

    if not summary:
        return None

    # Safeguard validation with retry (re-summarize from original, not fix)
    if config.mode == "safeguard":
        summary = await _safeguard_validate(
            summary=summary,
            identifiers=identifiers,
            provider=provider,
            max_retries=min(config.max_retries, 3),
            original_messages=stripped,
            latest_ask=_extract_latest_user_ask(summarize_msgs),
            identifier_policy=config.identifier_policy,
            last_successful_summary=session.latest_compaction_summary,
        )

    # Append tool failures section
    tool_failure_section = _format_tool_failures_section(tool_failures)
    if tool_failure_section:
        summary = _append_section(summary, tool_failure_section)

    # Append file operations sections
    file_ops_section = _format_file_operations(read_files, modified_files)
    if file_ops_section:
        summary = _append_section(summary, file_ops_section)

    # Create compaction entry
    entry = CompactionEntry(
        summary=summary,
        tokens_before=tokens_before,
        tokens_after=_estimate_tokens(summary),
        first_kept_entry_id=messages[-keep_count].id if keep_count <= len(messages) else None,
    )

    # Replace messages: compaction summary as system context + kept messages
    kept_messages = messages[-keep_count:]
    session.compaction_entries.append(entry)
    session.messages = kept_messages
    session._rewrite()

    return entry


def _append_section(summary: str, section: str) -> str:
    """Append a section to a summary, handling whitespace."""
    if not section:
        return summary
    if not summary.strip():
        return section.lstrip()
    return f"{summary}{section}"


def _chunk_messages(
    messages: list[AgentMessage], context_max_tokens: int
) -> list[list[AgentMessage]]:
    """Split messages into chunks that fit within token budget."""
    chunk_budget = int(context_max_tokens * BASE_CHUNK_RATIO) - OVERHEAD_TOKENS
    if chunk_budget < 2000:
        chunk_budget = 2000

    chunks: list[list[AgentMessage]] = []
    current_chunk: list[AgentMessage] = []
    current_tokens = 0

    for msg in messages:
        msg_tokens = _estimate_tokens(msg.text)
        for tu in msg.tool_uses:
            msg_tokens += _estimate_tokens(json.dumps(tu.input))
        for tr in msg.tool_results:
            msg_tokens += _estimate_tokens(tr.content)

        if current_tokens + msg_tokens > chunk_budget and current_chunk:
            chunks.append(current_chunk)
            current_chunk = []
            current_tokens = 0

        current_chunk.append(msg)
        current_tokens += msg_tokens

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


async def _safeguard_validate(
    summary: str,
    identifiers: list[str],
    provider: ModelProvider,
    max_retries: int,
    original_messages: list[AgentMessage],
    latest_ask: str | None,
    identifier_policy: str,
    last_successful_summary: str | None,
) -> str:
    """Validate summary quality and retry by re-summarizing from original messages.

    On safeguard failure, re-summarizes from original messages (not ask LLM to fix).
    Falls back to last successful summary on retry error.
    Max 3 retries.
    """
    capped_retries = min(max_retries, 3)

    for attempt in range(capped_retries + 1):
        ok, reasons = _audit_summary_quality(
            summary, identifiers, latest_ask, identifier_policy
        )

        if ok:
            return summary

        if attempt >= capped_retries:
            # Exhausted retries, return best effort
            log.warning(
                "Safeguard validation exhausted %d retries, reasons: %s",
                capped_retries, ", ".join(reasons),
            )
            break

        # Re-summarize from original messages (not ask LLM to fix the summary)
        log.info(
            "Safeguard retry %d/%d, re-summarizing from original messages. Reasons: %s",
            attempt + 1, capped_retries, ", ".join(reasons),
        )

        # Build quality feedback to inject into the re-summarization prompt
        quality_feedback = (
            f"Previous summary failed quality checks ({', '.join(reasons)}). "
            "Fix all issues and include every required section with exact identifiers preserved."
        )

        try:
            context_text = _messages_to_text(original_messages)
            prompt = _build_summarization_prompt(
                context_text, identifier_policy, identifiers, None
            )
            # Append quality feedback so the model knows what to fix
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
            # Fallback to last successful summary if available
            if last_successful_summary:
                log.info("Falling back to last successful summary.")
                return last_successful_summary
            # Otherwise return the current (imperfect) summary
            break

    return summary
