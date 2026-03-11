"""Multi-stage compaction with identifier preservation and safeguard validation.

Identifier extraction and safeguard validation are split into ``identifiers``
and ``safeguard`` submodules.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import TYPE_CHECKING

from openclaw.agent.types import (
    AgentMessage,
    CompactionEntry,
    TextBlock,
    ToolResultBlock,
)

# Re-export public API for backward compatibility
from openclaw.session.identifiers import (  # noqa: F401
    MAX_EXTRACTED_IDENTIFIERS,
    extract_identifiers,
)
from openclaw.session.safeguard import (  # noqa: F401
    ToolFailure,
    _has_required_sections_in_order,
    audit_summary_quality as _audit_summary_quality,
    collect_tool_failures,
    collect_file_operations as _collect_file_operations,
    extract_latest_user_ask as _extract_latest_user_ask,
    format_file_operations as _format_file_operations,
    format_tool_failures_section as _format_tool_failures_section,
    safeguard_validate as _safeguard_validate,
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
SAFETY_MARGIN = 1.2  # 20% buffer for estimate_tokens() inaccuracy
DEFAULT_PARTS = 3
DEFAULT_SUMMARY_FALLBACK = "No prior history."

# Merge prompt for combining partial summaries
MERGE_SUMMARIES_INSTRUCTIONS = "\n".join([
    "Merge these partial summaries into a single cohesive summary.",
    "",
    "MUST PRESERVE:",
    "- Active tasks and their current status (in-progress, blocked, pending)",
    "- Batch operation progress (e.g., '5/17 items completed')",
    "- The last thing the user requested and what was being done about it",
    "- Decisions made and their rationale",
    "- TODOs, open questions, and constraints",
    "- Any commitments or follow-ups promised",
    "",
    "PRIORITIZE recent context over older history. The agent needs to know",
    "what it was doing, not just what was discussed.",
])

MAX_LLM_RETRIES = 3
LLM_RETRY_BASE_DELAY = 1.0  # seconds, doubles each retry


# ---------------------------------------------------------------------------
# Token estimation helper (consolidated — used by multiple functions)
# ---------------------------------------------------------------------------

def _estimate_tokens(text: str) -> int:
    from openclaw.tokenizer import estimate_tokens
    return estimate_tokens(text)


def _estimate_message_tokens(msg: AgentMessage) -> int:
    """Estimate total tokens for a single message including tool uses/results."""
    total = _estimate_tokens(msg.text)
    for tu in msg.tool_uses:
        try:
            total += _estimate_tokens(json.dumps(tu.input))
        except Exception:
            total += 32
    for tr in msg.tool_results:
        total += _estimate_tokens(tr.content)
    return total


def _estimate_messages_tokens(messages: list[AgentMessage]) -> int:
    """Estimate total tokens across a list of messages."""
    return sum(_estimate_message_tokens(msg) for msg in messages)


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
# Checkpoint
# ---------------------------------------------------------------------------

def _write_checkpoint(messages: list[AgentMessage], workspace_dir: str) -> None:
    """Write a context checkpoint before compaction for recovery."""
    import datetime
    from pathlib import Path

    checkpoint_path = Path(workspace_dir) / ".context-checkpoint.md"
    now = datetime.datetime.now().isoformat()

    last_user_ask = ""
    for msg in reversed(messages):
        if msg.role == "user" and msg.text.strip():
            last_user_ask = msg.text.strip()[:500]
            break

    last_assistant = ""
    for msg in reversed(messages):
        if msg.role == "assistant" and msg.text.strip():
            last_assistant = msg.text.strip()[:500]
            break

    content = f"""# Context Checkpoint — {now}

## Last User Request
{last_user_ask or "(none)"}

## Last Assistant Response
{last_assistant or "(none)"}

## Session Stats
- Total messages: {len(messages)}
- Compaction triggered at: {now}
"""

    try:
        checkpoint_path.write_text(content, encoding="utf-8")
    except OSError:
        log.debug("Failed to write checkpoint to %s", checkpoint_path)


# ---------------------------------------------------------------------------
# Dynamic keep count
# ---------------------------------------------------------------------------

def _compute_keep_count(
    messages: list[AgentMessage],
    reserve_tokens_floor: int,
) -> int:
    """Dynamically compute how many recent messages to keep during compaction."""
    keep = 0
    budget = reserve_tokens_floor
    for msg in reversed(messages):
        msg_tokens = _estimate_message_tokens(msg)
        if keep >= 2 and msg_tokens > budget:
            break
        budget -= msg_tokens
        keep += 1
        if budget <= 0:
            break
    return max(2, keep)


# ---------------------------------------------------------------------------
# Orphan tool result cleanup
# ---------------------------------------------------------------------------


def _strip_leading_orphan_tool_results(
    messages: list[AgentMessage],
) -> list[AgentMessage]:
    """Remove orphan tool_results at the start of the kept message window.

    After compaction slices messages[-keep_count:], the first message may be
    a user message containing tool_results whose preceding assistant message
    (with matching tool_use) was discarded.  This violates the API invariant
    that every tool result must follow a tool_calls message.

    We strip tool_result blocks from leading user messages until we hit a
    message that has non-tool-result content or an assistant message.
    """
    from openclaw.agent.types import ToolResultBlock

    cleaned: list[AgentMessage] = []
    found_anchor = False

    for msg in messages:
        if found_anchor:
            cleaned.append(msg)
            continue

        # Once we find an assistant message, everything from here is safe
        if msg.role == "assistant":
            found_anchor = True
            cleaned.append(msg)
            continue

        # User message — check for orphan tool_results
        non_tool_blocks = [
            b for b in msg.content if not isinstance(b, ToolResultBlock)
        ]

        if non_tool_blocks:
            # Has real text content — keep non-tool blocks, drop tool_results
            msg.content = non_tool_blocks
            found_anchor = True
            cleaned.append(msg)
        else:
            # Entire message is tool_results with no anchor — drop it
            log.debug(
                "Stripped orphan tool_result message (id=%s) after compaction",
                msg.id,
            )

    return cleaned if cleaned else messages


# ---------------------------------------------------------------------------
# Main compaction flow
# ---------------------------------------------------------------------------

async def compact_session(
    session: SessionManager,
    provider: ModelProvider,
    config: CompactionConfig,
    context_max_tokens: int,
    workspace_dir: str = "",
    reserve_tokens_floor: int = 20000,
) -> CompactionEntry | None:
    """Perform multi-stage compaction on the session."""
    messages = session.messages

    if workspace_dir:
        _write_checkpoint(messages, workspace_dir)
    if len(messages) < 4:
        return None

    tokens_before = session.estimate_tokens()

    keep_count = _compute_keep_count(messages, reserve_tokens_floor)
    summarize_msgs = messages[:-keep_count] if len(messages) > keep_count else []

    if not summarize_msgs:
        return None

    stripped = strip_tool_result_details(summarize_msgs)

    from openclaw.session.identifiers import extract_identifiers_from_recent
    identifiers = (
        extract_identifiers_from_recent(stripped, max_messages=10)
        if config.identifier_policy != "off"
        else []
    )

    read_files, modified_files = _collect_file_operations(summarize_msgs)
    tool_failures = collect_tool_failures(summarize_msgs)

    chunk_ratio = compute_adaptive_chunk_ratio(stripped, context_max_tokens)

    previous_summary = session.latest_compaction_summary
    total_tokens = _estimate_messages_tokens(stripped)
    chunk_budget = int(context_max_tokens * chunk_ratio) - OVERHEAD_TOKENS

    if total_tokens > chunk_budget * 2:
        summary = await summarize_in_stages(
            stripped, provider, context_max_tokens,
            config.identifier_policy, identifiers,
            previous_summary=previous_summary,
            parts=DEFAULT_PARTS,
            chunk_ratio=chunk_ratio,
        )
    else:
        summary = await summarize_with_fallback(
            stripped, provider, context_max_tokens,
            config.identifier_policy, identifiers,
            previous_summary=previous_summary,
            chunk_ratio=chunk_ratio,
        )

    if not summary:
        return None

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

    tool_failure_section = _format_tool_failures_section(tool_failures)
    if tool_failure_section:
        summary = _append_section(summary, tool_failure_section)

    file_ops_section = _format_file_operations(read_files, modified_files)
    if file_ops_section:
        summary = _append_section(summary, file_ops_section)

    entry = CompactionEntry(
        summary=summary,
        tokens_before=tokens_before,
        tokens_after=_estimate_tokens(summary),
        first_kept_entry_id=messages[-keep_count].id if keep_count <= len(messages) else None,
    )

    kept_messages = messages[-keep_count:]

    # Ensure the first kept message doesn't have orphan tool_results
    # (i.e. tool_results whose preceding assistant tool_use was discarded)
    kept_messages = _strip_leading_orphan_tool_results(kept_messages)

    session.compaction_entries.append(entry)
    session.messages = kept_messages
    session._rewrite()

    return entry


def _append_section(summary: str, section: str) -> str:
    if not section:
        return summary
    if not summary.strip():
        return section.lstrip()
    return f"{summary}{section}"


# ---------------------------------------------------------------------------
# Chunking helpers
# ---------------------------------------------------------------------------

def _normalize_parts(parts: int, message_count: int) -> int:
    if parts <= 1:
        return 1
    return min(max(1, parts), max(1, message_count))


def split_messages_by_token_share(
    messages: list[AgentMessage],
    parts: int = DEFAULT_PARTS,
) -> list[list[AgentMessage]]:
    """Split messages into N parts where each part has roughly equal token share."""
    if not messages:
        return []
    normalized_parts = _normalize_parts(parts, len(messages))
    if normalized_parts <= 1:
        return [messages]

    total_tokens = _estimate_messages_tokens(messages)
    target_tokens = total_tokens / normalized_parts
    chunks: list[list[AgentMessage]] = []
    current: list[AgentMessage] = []
    current_tokens = 0

    for msg in messages:
        msg_tokens = _estimate_message_tokens(msg)

        if (
            len(chunks) < normalized_parts - 1
            and current
            and current_tokens + msg_tokens > target_tokens
        ):
            chunks.append(current)
            current = []
            current_tokens = 0

        current.append(msg)
        current_tokens += msg_tokens

    if current:
        chunks.append(current)

    return chunks


def compute_adaptive_chunk_ratio(
    messages: list[AgentMessage],
    context_window: int,
) -> float:
    """Compute adaptive chunk ratio based on average message size."""
    if not messages:
        return BASE_CHUNK_RATIO

    total_tokens = _estimate_messages_tokens(messages)
    avg_tokens = total_tokens / len(messages)

    safe_avg_tokens = avg_tokens * SAFETY_MARGIN
    avg_ratio = safe_avg_tokens / context_window

    if avg_ratio > 0.1:
        reduction = min(avg_ratio * 2, BASE_CHUNK_RATIO - MIN_CHUNK_RATIO)
        return max(MIN_CHUNK_RATIO, BASE_CHUNK_RATIO - reduction)

    return BASE_CHUNK_RATIO


def is_oversized_for_summary(msg: AgentMessage, context_window: int) -> bool:
    """Check if a single message is too large to summarize."""
    msg_tokens = _estimate_message_tokens(msg)
    return (msg_tokens * SAFETY_MARGIN) > context_window * 0.5


# ---------------------------------------------------------------------------
# Multi-stage summarization
# ---------------------------------------------------------------------------

async def summarize_with_fallback(
    messages: list[AgentMessage],
    provider: ModelProvider,
    context_max_tokens: int,
    identifier_policy: str,
    identifiers: list[str],
    previous_summary: str | None = None,
    chunk_ratio: float = BASE_CHUNK_RATIO,
) -> str:
    """Summarize with progressive fallback for oversized messages."""
    if not messages:
        return previous_summary or DEFAULT_SUMMARY_FALLBACK

    chunk_budget = int(context_max_tokens * chunk_ratio) - OVERHEAD_TOKENS
    if chunk_budget < 2000:
        chunk_budget = 2000

    try:
        return await _summarize_chunks(
            messages, provider, chunk_budget, identifier_policy, identifiers, previous_summary
        )
    except Exception as full_error:
        log.warning("Full summarization failed, trying partial: %s", full_error)

    small_messages: list[AgentMessage] = []
    oversized_notes: list[str] = []

    for msg in messages:
        if is_oversized_for_summary(msg, context_max_tokens):
            tokens = _estimate_tokens(msg.text)
            oversized_notes.append(
                f"[Large {msg.role} (~{round(tokens / 1000)}K tokens) omitted from summary]"
            )
        else:
            small_messages.append(msg)

    if small_messages:
        try:
            partial_summary = await _summarize_chunks(
                small_messages, provider, chunk_budget,
                identifier_policy, identifiers, previous_summary,
            )
            notes = f"\n\n{chr(10).join(oversized_notes)}" if oversized_notes else ""
            return partial_summary + notes
        except Exception as partial_error:
            log.warning("Partial summarization also failed: %s", partial_error)

    return (
        f"Context contained {len(messages)} messages "
        f"({len(oversized_notes)} oversized). "
        f"Summary unavailable due to size limits."
    )


async def _summarize_chunks(
    messages: list[AgentMessage],
    provider: ModelProvider,
    chunk_budget: int,
    identifier_policy: str,
    identifiers: list[str],
    previous_summary: str | None = None,
) -> str:
    """Sequential chunk summarization — each chunk summary feeds into the next."""
    if not messages:
        return previous_summary or DEFAULT_SUMMARY_FALLBACK

    effective_budget = max(1, int(chunk_budget / SAFETY_MARGIN))

    chunks: list[list[AgentMessage]] = []
    current_chunk: list[AgentMessage] = []
    current_tokens = 0

    for msg in messages:
        msg_tokens = _estimate_message_tokens(msg)

        if current_chunk and current_tokens + msg_tokens > effective_budget:
            chunks.append(current_chunk)
            current_chunk = []
            current_tokens = 0

        current_chunk.append(msg)
        current_tokens += msg_tokens

    if current_chunk:
        chunks.append(current_chunk)

    summary = previous_summary
    for chunk in chunks:
        chunk_text = _messages_to_text(chunk)
        prompt = _build_summarization_prompt(
            chunk_text, identifier_policy, identifiers, summary,
        )
        summary = await _llm_complete_with_retry(
            provider,
            system="You are a precise summarizer. Follow the structure exactly.",
            messages=[AgentMessage(role="user", content=[TextBlock(text=prompt)])],
        )

    return summary or DEFAULT_SUMMARY_FALLBACK


async def summarize_in_stages(
    messages: list[AgentMessage],
    provider: ModelProvider,
    context_max_tokens: int,
    identifier_policy: str,
    identifiers: list[str],
    previous_summary: str | None = None,
    parts: int = DEFAULT_PARTS,
    min_messages_for_split: int = 4,
    chunk_ratio: float = BASE_CHUNK_RATIO,
) -> str:
    """Multi-stage compaction: split → summarize each → merge partial summaries."""
    if not messages:
        return previous_summary or DEFAULT_SUMMARY_FALLBACK

    min_messages_for_split = max(2, min_messages_for_split)
    normalized_parts = _normalize_parts(parts, len(messages))
    total_tokens = _estimate_messages_tokens(messages)

    chunk_budget = int(context_max_tokens * chunk_ratio) - OVERHEAD_TOKENS
    if chunk_budget < 2000:
        chunk_budget = 2000

    if (
        normalized_parts <= 1
        or len(messages) < min_messages_for_split
        or total_tokens <= chunk_budget
    ):
        return await summarize_with_fallback(
            messages, provider, context_max_tokens,
            identifier_policy, identifiers, previous_summary, chunk_ratio,
        )

    splits = [
        chunk for chunk in split_messages_by_token_share(messages, normalized_parts)
        if chunk
    ]
    if len(splits) <= 1:
        return await summarize_with_fallback(
            messages, provider, context_max_tokens,
            identifier_policy, identifiers, previous_summary, chunk_ratio,
        )

    partial_summaries: list[str] = []
    for chunk in splits:
        partial = await summarize_with_fallback(
            chunk, provider, context_max_tokens,
            identifier_policy, identifiers,
            previous_summary=None, chunk_ratio=chunk_ratio,
        )
        partial_summaries.append(partial)

    if len(partial_summaries) == 1:
        return partial_summaries[0]

    merge_prompt = (
        f"{MERGE_SUMMARIES_INSTRUCTIONS}\n\n"
        "Partial summaries to merge:\n\n"
        + "\n\n---\n\n".join(partial_summaries)
    )

    merged = await _llm_complete_with_retry(
        provider,
        system="You are a precise summarizer. Follow the structure exactly.",
        messages=[AgentMessage(role="user", content=[TextBlock(text=merge_prompt)])],
    )

    return merged or DEFAULT_SUMMARY_FALLBACK


# ---------------------------------------------------------------------------
# Backward compatibility aliases
# ---------------------------------------------------------------------------

# These private names are imported by eval_intelligence.py and test_live.py
_extract_identifiers_from_recent = lambda msgs, max_messages=10: (
    __import__('openclaw.session.identifiers', fromlist=['extract_identifiers_from_recent'])
    .extract_identifiers_from_recent(msgs, max_messages)
)
_audit_summary_quality_compat = _audit_summary_quality
