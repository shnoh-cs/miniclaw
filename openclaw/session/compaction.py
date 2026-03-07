"""Multi-stage compaction with identifier preservation and safeguard validation."""

from __future__ import annotations

import json
import re
import time
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

# Chunk ratios for multi-stage summarization
BASE_CHUNK_RATIO = 0.4
MIN_CHUNK_RATIO = 0.15
OVERHEAD_TOKENS = 4096

# Identifier patterns to preserve during compaction
_IDENTIFIER_PATTERNS = [
    re.compile(r"[0-9a-fA-F]{8,}"),  # hex strings
    re.compile(r"https?://\S+"),  # URLs
    re.compile(r"\b[\w.-]+\.(ts|py|js|go|rs|md|json|yaml|toml)\b"),  # file paths
    re.compile(r"\b\d{1,3}(\.\d{1,3}){3}\b"),  # IPs
    re.compile(r":\d{2,5}\b"),  # ports
    re.compile(r"[A-Za-z0-9_-]{20,}"),  # long tokens/API keys
]

# Required sections in a safeguard-validated summary
_REQUIRED_SECTIONS = [
    "## Decisions",
    "## Open TODOs",
    "## Constraints",
    "## Pending",
    "## Exact identifiers",
]


def extract_identifiers(text: str) -> set[str]:
    """Extract identifiers (UUIDs, hashes, URLs, IPs, ports, file names) from text."""
    ids: set[str] = set()
    for pattern in _IDENTIFIER_PATTERNS:
        for match in pattern.finditer(text):
            ids.add(match.group(0))
    return ids


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
    identifiers: set[str],
    previous_summary: str | None = None,
) -> str:
    """Build the summarization instruction prompt."""
    prompt = (
        "Summarize the following conversation into a compact summary. "
        "Preserve the most important information for continuing the work.\n\n"
        "Structure your summary with these sections:\n"
        "## Decisions\nKey decisions made.\n"
        "## Open TODOs\nTasks that are still pending.\n"
        "## Constraints\nRules or constraints discovered.\n"
        "## Pending user asks\nRequests from the user not yet addressed.\n"
        "## Exact identifiers\nCritical IDs, hashes, URLs, file paths.\n\n"
    )

    if identifier_policy == "strict" and identifiers:
        id_list = "\n".join(f"- {id_}" for id_ in sorted(identifiers)[:50])
        prompt += (
            f"IMPORTANT: You MUST preserve these exact identifiers:\n{id_list}\n\n"
        )

    if previous_summary:
        prompt += f"Previous summary to incorporate:\n{previous_summary}\n\n"

    prompt += f"Conversation to summarize:\n{context_text}"
    return prompt


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

    # Extract identifiers for preservation
    all_text = _messages_to_text(stripped)
    identifiers = extract_identifiers(all_text) if config.identifier_policy != "off" else set()

    # Multi-stage chunking if context is too large
    chunks = _chunk_messages(stripped, context_max_tokens)
    previous_summary = session.latest_compaction_summary

    summary = ""
    for chunk in chunks:
        chunk_text = _messages_to_text(chunk)
        prompt = _build_summarization_prompt(
            chunk_text, config.identifier_policy, identifiers, previous_summary
        )

        summary = await provider.complete(
            system="You are a precise summarizer. Follow the structure exactly.",
            messages=[AgentMessage(role="user", content=[TextBlock(text=prompt)])],
            model=None,  # uses compaction model
        )
        previous_summary = summary

    if not summary:
        return None

    # Safeguard validation
    if config.mode == "safeguard":
        summary = await _safeguard_validate(
            summary, identifiers, provider, config.max_retries
        )

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
    identifiers: set[str],
    provider: ModelProvider,
    max_retries: int,
) -> str:
    """Validate summary quality and retry if sections or identifiers are missing."""
    for attempt in range(max_retries):
        issues: list[str] = []

        # Check required sections
        for section in _REQUIRED_SECTIONS:
            if section.lower() not in summary.lower():
                issues.append(f"Missing section: {section}")

        # Check identifier preservation
        if identifiers:
            found = extract_identifiers(summary)
            missing = identifiers - found
            # Only flag if more than 30% are missing
            if len(missing) > len(identifiers) * 0.3:
                sample = sorted(missing)[:10]
                issues.append(
                    f"Missing {len(missing)}/{len(identifiers)} identifiers, "
                    f"e.g.: {', '.join(sample)}"
                )

        if not issues:
            return summary

        # Retry with feedback
        feedback = "The summary has quality issues:\n" + "\n".join(f"- {i}" for i in issues)
        feedback += f"\n\nOriginal summary:\n{summary}\n\nPlease fix and return an improved summary."

        summary = await provider.complete(
            system="You are a precise summarizer. Fix the issues listed.",
            messages=[AgentMessage(role="user", content=[TextBlock(text=feedback)])],
        )

    return summary  # return best effort after exhausting retries
