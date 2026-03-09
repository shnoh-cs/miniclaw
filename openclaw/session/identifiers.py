"""Identifier extraction, sanitization, and normalization for compaction."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from openclaw.agent.types import AgentMessage

# Maximum identifiers to preserve during compaction
MAX_EXTRACTED_IDENTIFIERS = 12

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


def extract_identifiers_from_recent(messages: list[AgentMessage], max_messages: int = 10) -> list[str]:
    """Extract identifiers from the last N messages only."""
    from openclaw.session.compaction import _messages_to_text
    recent = messages[-max_messages:] if len(messages) > max_messages else messages
    text = _messages_to_text(recent)
    return extract_identifiers(text)
