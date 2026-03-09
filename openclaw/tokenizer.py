"""Shared token estimation using tiktoken.

Provides accurate token counts instead of the naive ``len(text) // 4``
heuristic, which significantly underestimates for CJK text (Korean, Chinese,
Japanese) where each character often maps to 2-3 tokens.

Falls back to ``len(text) // 3`` if tiktoken is unavailable.
"""

from __future__ import annotations

import functools
import logging

log = logging.getLogger("openclaw.tokenizer")

_FALLBACK_CHARS_PER_TOKEN = 3  # conservative fallback (better than //4 for CJK)


@functools.lru_cache(maxsize=1)
def _get_encoding():
    """Load the tiktoken encoding (cached singleton)."""
    try:
        import tiktoken
        return tiktoken.get_encoding("cl100k_base")
    except Exception:
        return None


def estimate_tokens(text: str) -> int:
    """Estimate token count for *text* using tiktoken.

    Uses cl100k_base encoding (GPT-4 / Claude-compatible BPE).
    Falls back to ``len(text) // 3`` if tiktoken is unavailable.
    """
    if not text:
        return 0
    enc = _get_encoding()
    if enc is not None:
        try:
            return len(enc.encode(text, disallowed_special=()))
        except Exception:
            pass
    return len(text) // _FALLBACK_CHARS_PER_TOKEN
