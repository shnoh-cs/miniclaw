"""Thinking level management with automatic fallback."""

from __future__ import annotations

from openclaw.agent.types import ThinkingLevel


def resolve_thinking(
    requested: ThinkingLevel,
    model_supported: set[ThinkingLevel] | None = None,
) -> ThinkingLevel:
    """Resolve the effective thinking level with fallback.

    If the model doesn't support the requested level, fall back to the
    next lower level until we find one that's supported (or OFF).
    """
    if model_supported is None:
        # Assume all levels supported if not specified
        return requested

    level = requested
    while level != ThinkingLevel.OFF:
        if level in model_supported:
            return level
        level = level.fallback()

    return ThinkingLevel.OFF


def parse_thinking_directive(text: str) -> ThinkingLevel | None:
    """Parse a /think directive from message text.

    Supports: /t <level>, /think:<level>, /thinking <level>
    Returns None if no directive found.
    """
    import re

    patterns = [
        r"/t\s+(\w[\w\s]*)",
        r"/think:(\w+)",
        r"/thinking\s+(\w[\w\s]*)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return ThinkingLevel.from_str(match.group(1).strip())
    return None


def strip_thinking_directive(text: str) -> str:
    """Remove thinking directives from message text."""
    import re

    patterns = [
        r"/t\s+\w[\w\s]*",
        r"/think:\w+",
        r"/thinking\s+\w[\w\s]*",
    ]
    result = text
    for pattern in patterns:
        result = re.sub(pattern, "", result, flags=re.IGNORECASE)
    return result.strip()
