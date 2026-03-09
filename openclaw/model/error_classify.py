"""Error classification for model failover.

Pattern tables and classifier ported from OpenClaw TypeScript
failover-error / failover-matches modules.
"""

from __future__ import annotations

import re

from openclaw.agent.types import FailoverReason

# ---------------------------------------------------------------------------
# Error pattern tables (ported from failover-matches.ts)
# ---------------------------------------------------------------------------

_ErrorPattern = re.Pattern[str] | str


def _matches(text: str, patterns: list[_ErrorPattern]) -> bool:
    """Return True if *text* matches any pattern (case-insensitive substring
    for plain strings, regex match for compiled patterns)."""
    if not text:
        return False
    lower = text.lower()
    for p in patterns:
        if isinstance(p, re.Pattern):
            if p.search(lower):
                return True
        elif lower.find(p) != -1:
            return True
    return False


# -- Rate limit ---------------------------------------------------------------
_RATE_LIMIT_PATTERNS: list[_ErrorPattern] = [
    re.compile(r"rate[_ ]limit|too many requests|429"),
    "model_cooldown",
    "exceeded your current quota",
    "resource has been exhausted",
    "quota exceeded",
    "resource_exhausted",
    "usage limit",
    re.compile(r"\btpm\b", re.IGNORECASE),
    "tokens per minute",
]

_PERIODIC_USAGE_LIMIT_RE = re.compile(
    r"\b(?:daily|weekly|monthly)(?:/(?:daily|weekly|monthly))* (?:usage )?limit(?:s)?"
    r"(?: (?:exhausted|reached|exceeded))?\b",
    re.IGNORECASE,
)

# -- Overloaded ---------------------------------------------------------------
_OVERLOADED_PATTERNS: list[_ErrorPattern] = [
    re.compile(r'overloaded_error|"type"\s*:\s*"overloaded_error"', re.IGNORECASE),
    "overloaded",
    re.compile(
        r"service[_ ]unavailable.*(?:overload|capacity|high[_ ]demand)"
        r"|(?:overload|capacity|high[_ ]demand).*service[_ ]unavailable",
        re.IGNORECASE,
    ),
    "high demand",
]

# -- Timeout / transient ------------------------------------------------------
_TIMEOUT_PATTERNS: list[_ErrorPattern] = [
    "timeout",
    "timed out",
    "service unavailable",
    "deadline exceeded",
    "context deadline exceeded",
    "connection error",
    "network error",
    "network request failed",
    "fetch failed",
    "socket hang up",
    re.compile(r"\beconn(?:refused|reset|aborted)\b", re.IGNORECASE),
    re.compile(r"\betimedout\b", re.IGNORECASE),
    re.compile(r"\benotfound\b", re.IGNORECASE),
    re.compile(r"\beai_again\b", re.IGNORECASE),
    re.compile(r"without sending (?:any )?chunks?", re.IGNORECASE),
    re.compile(r"\bstop reason:\s*(?:abort|error)\b", re.IGNORECASE),
    re.compile(r"\breason:\s*(?:abort|error)\b", re.IGNORECASE),
    re.compile(r"\bunhandled stop reason:\s*(?:abort|error)\b", re.IGNORECASE),
]

# -- Billing ------------------------------------------------------------------
_BILLING_PATTERNS: list[_ErrorPattern] = [
    re.compile(
        r"""["']?(?:status|code)["']?\s*[:=]\s*402\b"""
        r"""|\bhttp\s*402\b"""
        r"""|\berror(?:\s+code)?\s*[:=]?\s*402\b"""
        r"""|\b(?:got|returned|received)\s+(?:a\s+)?402\b"""
        r"""|^\s*402\s+payment""",
        re.IGNORECASE,
    ),
    "payment required",
    "insufficient credits",
    re.compile(r"insufficient[_ ]quota", re.IGNORECASE),
    "credit balance",
    "plans & billing",
    "insufficient balance",
]

# -- Auth (permanent) ---------------------------------------------------------
_AUTH_PERMANENT_PATTERNS: list[_ErrorPattern] = [
    re.compile(r"api[_ ]?key[_ ]?(?:revoked|invalid|deactivated|deleted)", re.IGNORECASE),
    "invalid_api_key",
    "key has been disabled",
    "key has been revoked",
    "account has been deactivated",
    re.compile(r"could not (?:authenticate|validate).*(?:api[_ ]?key|credentials)", re.IGNORECASE),
    "permission_error",
    "not allowed for this organization",
]

# -- Auth (transient) ---------------------------------------------------------
_AUTH_PATTERNS: list[_ErrorPattern] = [
    re.compile(r"invalid[_ ]?api[_ ]?key"),
    "incorrect api key",
    "invalid token",
    "authentication",
    "re-authenticate",
    "oauth token refresh failed",
    "unauthorized",
    "forbidden",
    "access denied",
    "insufficient permissions",
    "insufficient permission",
    re.compile(r"missing scopes?:", re.IGNORECASE),
    "expired",
    "token has expired",
    re.compile(r"\b401\b"),
    re.compile(r"\b403\b"),
    "no credentials found",
    "no api key found",
]

# -- Format -------------------------------------------------------------------
_FORMAT_PATTERNS: list[_ErrorPattern] = [
    "string should match pattern",
    "tool_use.id",
    "tool_use_id",
    "messages.1.content.1.tool_use.id",
    "invalid request format",
    re.compile(r"tool call id was.*must be", re.IGNORECASE),
]

# -- Context overflow ---------------------------------------------------------
_CONTEXT_OVERFLOW_RE = re.compile(
    r"(context.?length|token.?limit|maximum.?context|too.?long"
    r"|exceed.*(length|limit|tokens)"
    r"|request_too_large|prompt is too long"
    r"|context.?window.?exceeded"
    r"|model.?token.?limit)",
    re.IGNORECASE,
)

# -- Model not found ----------------------------------------------------------
_MODEL_NOT_FOUND_PATTERNS: list[_ErrorPattern] = [
    "unknown model",
    "model not found",
    "model_not_found",
    "not_found_error",
    re.compile(r"does not exist.*model|model.*does not exist", re.IGNORECASE),
    re.compile(r"invalid model(?! reference)", re.IGNORECASE),
    re.compile(r"models/[^\s]+ is not found", re.IGNORECASE),
]

# -- Session expired ----------------------------------------------------------
_SESSION_EXPIRED_PATTERNS: list[_ErrorPattern] = [
    "session not found",
    "session does not exist",
    "session expired",
    "session invalid",
    "conversation not found",
    "conversation does not exist",
    "conversation expired",
    "conversation invalid",
    "no such session",
    "invalid session",
    "session id not found",
    "conversation id not found",
]

# -- Transient HTTP status codes ----------------------------------------------
_TRANSIENT_HTTP_CODES = {500, 502, 503, 504, 521, 522, 523, 524, 529}

_HTTP_STATUS_RE = re.compile(r"^(?:http\s*)?(\d{3})(?:\s+|$)", re.IGNORECASE)


# ---------------------------------------------------------------------------
# HTTP-status-based classification
# ---------------------------------------------------------------------------

def _classify_from_http_status(status: int | None, text: str) -> FailoverReason | None:
    """Classify based on an explicit HTTP status code."""
    if status is None:
        return None

    if status == 402:
        lower = text.lower()
        has_temporary = any(w in lower for w in ("try again", "retry", "temporary", "cooldown"))
        has_usage = any(w in lower for w in ("usage limit", "rate limit", "organization usage"))
        if has_temporary and has_usage:
            return FailoverReason.RATE_LIMIT
        return FailoverReason.BILLING

    if status == 429:
        return FailoverReason.RATE_LIMIT
    if status in (401, 403):
        if _matches(text, _AUTH_PERMANENT_PATTERNS):
            return FailoverReason.AUTH_PERMANENT
        return FailoverReason.AUTH
    if status == 408:
        return FailoverReason.TIMEOUT
    if status == 503:
        if _matches(text, _OVERLOADED_PATTERNS):
            return FailoverReason.OVERLOADED
        return FailoverReason.TIMEOUT
    if status in (502, 504):
        return FailoverReason.TIMEOUT
    if status == 529:
        return FailoverReason.OVERLOADED
    if status == 400:
        if _matches(text, _BILLING_PATTERNS):
            return FailoverReason.BILLING
        return FailoverReason.FORMAT
    return None


def _extract_http_status(text: str) -> int | None:
    """Try to extract a leading HTTP status code from an error string."""
    m = _HTTP_STATUS_RE.match(text.strip())
    if m:
        code = int(m.group(1))
        if 100 <= code < 600:
            return code
    return None


# ---------------------------------------------------------------------------
# Main classifier
# ---------------------------------------------------------------------------

def classify_error(error: str | Exception) -> FailoverReason:
    """Classify an error string/exception into a ``FailoverReason``.

    The check order mirrors the TypeScript ``classifyFailoverReason`` cascade.
    """
    text = str(error)

    # 1. Try HTTP status code first
    status = _extract_http_status(text)
    status_reason = _classify_from_http_status(status, text)
    if status_reason is not None:
        return status_reason

    # 2. Session expired (narrow, check early)
    if _matches(text, _SESSION_EXPIRED_PATTERNS):
        return FailoverReason.SESSION_EXPIRED

    # 3. Model not found
    if _matches(text, _MODEL_NOT_FOUND_PATTERNS):
        return FailoverReason.MODEL_NOT_FOUND

    # 4. Periodic usage limit
    if _PERIODIC_USAGE_LIMIT_RE.search(text):
        return FailoverReason.BILLING if _matches(text, _BILLING_PATTERNS) else FailoverReason.RATE_LIMIT

    # 5. Rate limit
    if _matches(text, _RATE_LIMIT_PATTERNS):
        return FailoverReason.RATE_LIMIT

    # 6. Overloaded
    if _matches(text, _OVERLOADED_PATTERNS):
        return FailoverReason.OVERLOADED

    # 7. Transient HTTP errors
    if status is not None and status in _TRANSIENT_HTTP_CODES:
        if status == 529:
            return FailoverReason.OVERLOADED
        return FailoverReason.TIMEOUT

    # 8. JSON-wrapped internal server error
    lower = text.lower()
    if '"type":"api_error"' in lower and "internal server error" in lower:
        return FailoverReason.TIMEOUT

    # 9. Format errors
    if _matches(text, _FORMAT_PATTERNS):
        return FailoverReason.FORMAT

    # 10. Billing
    if _matches(text, _BILLING_PATTERNS):
        return FailoverReason.BILLING

    # 11. Timeout / transient network
    if _matches(text, _TIMEOUT_PATTERNS):
        return FailoverReason.TIMEOUT

    # 12. Context overflow
    if _CONTEXT_OVERFLOW_RE.search(text):
        return FailoverReason.CONTEXT_OVERFLOW

    # 13. Auth permanent
    if _matches(text, _AUTH_PERMANENT_PATTERNS):
        return FailoverReason.AUTH_PERMANENT

    # 14. Auth (transient)
    if _matches(text, _AUTH_PATTERNS):
        return FailoverReason.AUTH

    return FailoverReason.UNKNOWN


# ---------------------------------------------------------------------------
# Failover decision
# ---------------------------------------------------------------------------

_NO_FAILOVER_REASONS = frozenset({FailoverReason.CONTEXT_OVERFLOW})


def should_failover(reason: FailoverReason) -> bool:
    """Determine if the error warrants failover to another model/profile."""
    if reason == FailoverReason.UNKNOWN:
        return False
    return reason not in _NO_FAILOVER_REASONS
