"""Prompt injection defense: sanitize untrusted data before prompt injection."""

from __future__ import annotations

import re
import unicodedata

# Unicode categories to strip: control (Cc), format (Cf), line/paragraph separators
_STRIP_CATEGORIES = {"Cc", "Cf"}
_STRIP_CODEPOINTS = {0x2028, 0x2029}  # line/paragraph separators


def sanitize_text(text: str) -> str:
    """Strip dangerous Unicode characters from text (lossy).

    Removes control characters, format characters, and separators
    that could break prompt structure.
    """
    chars = []
    for ch in text:
        cat = unicodedata.category(ch)
        if cat in _STRIP_CATEGORIES and ch not in ("\n", "\r", "\t"):
            continue
        if ord(ch) in _STRIP_CODEPOINTS:
            continue
        chars.append(ch)
    return "".join(chars)


def escape_html_in_untrusted(text: str) -> str:
    """Escape < and > in untrusted text to prevent tag injection."""
    return text.replace("<", "&lt;").replace(">", "&gt;")


def wrap_untrusted(text: str, label: str = "untrusted", max_chars: int = 0) -> str:
    """Wrap untrusted data in <untrusted-text> tags with sanitization.

    Args:
        text: The untrusted text to wrap.
        label: Label for the untrusted block.
        max_chars: Maximum characters (0 = unlimited).
    """
    cleaned = sanitize_text(text)
    escaped = escape_html_in_untrusted(cleaned)
    if max_chars > 0 and len(escaped) > max_chars:
        escaped = escaped[:max_chars] + "\n...[truncated]"
    return f'<untrusted-text label="{label}">\n{escaped}\n</untrusted-text>'
