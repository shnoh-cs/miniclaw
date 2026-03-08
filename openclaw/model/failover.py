"""Error classification and model failover logic.

Mirrors the OpenClaw TypeScript failover-error / auth-profiles / failover-matches
modules so that error classification, cooldown management, retry guards and
overload pacing all behave identically.
"""

from __future__ import annotations

import fcntl
import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

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

# -- Transient HTTP status codes that indicate retryable transport issues ------
_TRANSIENT_HTTP_CODES = {500, 502, 503, 504, 521, 522, 523, 524, 529}

_HTTP_STATUS_RE = re.compile(r"^(?:http\s*)?(\d{3})(?:\s+|$)", re.IGNORECASE)


# ---------------------------------------------------------------------------
# HTTP-status-based classification
# ---------------------------------------------------------------------------

def _classify_from_http_status(status: int | None, text: str) -> FailoverReason | None:
    """Classify based on an explicit HTTP status code (mirrors
    ``classifyFailoverReasonFromHttpStatus`` in the TS codebase)."""
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

    The check order mirrors the TypeScript ``classifyFailoverReason`` cascade so
    that more-specific categories (session_expired, model_not_found) are tested
    before broader ones (auth, timeout).
    """
    text = str(error)

    # 1. Try HTTP status code first (if embedded in the text).
    status = _extract_http_status(text)
    status_reason = _classify_from_http_status(status, text)
    if status_reason is not None:
        return status_reason

    # 2. Session expired (narrow, check early).
    if _matches(text, _SESSION_EXPIRED_PATTERNS):
        return FailoverReason.SESSION_EXPIRED

    # 3. Model not found.
    if _matches(text, _MODEL_NOT_FOUND_PATTERNS):
        return FailoverReason.MODEL_NOT_FOUND

    # 4. Periodic usage limit (daily/weekly/monthly) -> billing or rate_limit.
    if _PERIODIC_USAGE_LIMIT_RE.search(text):
        return FailoverReason.BILLING if _matches(text, _BILLING_PATTERNS) else FailoverReason.RATE_LIMIT

    # 5. Rate limit.
    if _matches(text, _RATE_LIMIT_PATTERNS):
        return FailoverReason.RATE_LIMIT

    # 6. Overloaded.
    if _matches(text, _OVERLOADED_PATTERNS):
        return FailoverReason.OVERLOADED

    # 7. Transient HTTP errors (500/502/503/504/521-524/529).
    if status is not None and status in _TRANSIENT_HTTP_CODES:
        if status == 529:
            return FailoverReason.OVERLOADED
        return FailoverReason.TIMEOUT

    # 8. JSON-wrapped internal server error (Anthropic-style).
    lower = text.lower()
    if '"type":"api_error"' in lower and "internal server error" in lower:
        return FailoverReason.TIMEOUT

    # 9. Format errors.
    if _matches(text, _FORMAT_PATTERNS):
        return FailoverReason.FORMAT

    # 10. Billing.
    if _matches(text, _BILLING_PATTERNS):
        return FailoverReason.BILLING

    # 11. Timeout / transient network.
    if _matches(text, _TIMEOUT_PATTERNS):
        return FailoverReason.TIMEOUT

    # 12. Context overflow.
    if _CONTEXT_OVERFLOW_RE.search(text):
        return FailoverReason.CONTEXT_OVERFLOW

    # 13. Auth permanent (before generic auth).
    if _matches(text, _AUTH_PERMANENT_PATTERNS):
        return FailoverReason.AUTH_PERMANENT

    # 14. Auth (transient).
    if _matches(text, _AUTH_PATTERNS):
        return FailoverReason.AUTH

    return FailoverReason.UNKNOWN


# ---------------------------------------------------------------------------
# Failover decision
# ---------------------------------------------------------------------------

# Context overflow is NOT failover-worthy (needs compaction, not rotation).
_NO_FAILOVER_REASONS = frozenset({FailoverReason.CONTEXT_OVERFLOW})


def should_failover(reason: FailoverReason) -> bool:
    """Determine if the error warrants failover to another model/profile.

    Every classified reason triggers failover *except* CONTEXT_OVERFLOW (which
    requires compaction, not profile/model rotation).
    """
    if reason == FailoverReason.UNKNOWN:
        return False
    return reason not in _NO_FAILOVER_REASONS


# ---------------------------------------------------------------------------
# Auth profile cooldown tracking
# ---------------------------------------------------------------------------

# Transient: 1m -> 5m -> 25m -> 1h (cap)
_TRANSIENT_BACKOFF_SECONDS = [60, 300, 1500, 3600]
# Billing / auth_permanent: 5h -> 10h -> 24h (cap)
_BILLING_BACKOFF_SECONDS = [18000, 36000, 86400]


@dataclass
class ProfileCooldown:
    """Tracks cooldown state for an auth profile."""

    error_count: int = 0
    billing_error_count: int = 0
    cooldown_until: float = 0.0
    last_success: float = 0.0
    last_failure_at: float = 0.0

    @property
    def is_in_cooldown(self) -> bool:
        return time.time() < self.cooldown_until

    def mark_failure(self, reason: FailoverReason) -> None:
        """Record a failure.

        Timeout errors trigger rotation but do NOT increment error counters or
        set cooldown windows (the timeout is model/network-specific, not an
        auth-profile issue).

        If the profile is already in an active cooldown window, the window is
        NOT extended further (immutable active windows).
        """
        now = time.time()
        self.last_failure_at = now

        # Timeouts: rotate but don't mark the profile as failed.
        if reason == FailoverReason.TIMEOUT:
            return

        # Immutable active cooldown window: if already cooling down, only
        # increment counters but do not extend the window.
        already_cooling = self.is_in_cooldown

        if reason in (FailoverReason.BILLING, FailoverReason.AUTH_PERMANENT):
            idx = min(self.billing_error_count, len(_BILLING_BACKOFF_SECONDS) - 1)
            backoff = _BILLING_BACKOFF_SECONDS[idx]
            self.billing_error_count += 1
        else:
            idx = min(self.error_count, len(_TRANSIENT_BACKOFF_SECONDS) - 1)
            backoff = _TRANSIENT_BACKOFF_SECONDS[idx]
            self.error_count += 1

        if not already_cooling:
            self.cooldown_until = now + backoff

    def mark_success(self) -> None:
        self.error_count = 0
        self.billing_error_count = 0
        self.cooldown_until = 0.0
        self.last_success = time.time()

    def clear_if_expired(self) -> bool:
        """Clear cooldown and reset counters if the cooldown window has passed.

        Returns True if the cooldown was cleared.
        """
        if self.cooldown_until > 0 and time.time() >= self.cooldown_until:
            self.cooldown_until = 0.0
            self.error_count = 0
            self.billing_error_count = 0
            return True
        return False

    def reset_if_stale(self, stale_hours: int = 24) -> None:
        """Reset counters if last success was recent enough."""
        if self.last_success and (time.time() - self.last_success) < stale_hours * 3600:
            self.error_count = 0
            self.billing_error_count = 0


# ---------------------------------------------------------------------------
# API key rotation
# ---------------------------------------------------------------------------


@dataclass
class ApiKeyRotator:
    """Manages multiple API keys for a single provider with round-robin
    rotation and per-key cooldown tracking.

    Usage::

        rotator = ApiKeyRotator(keys=["sk-aaa", "sk-bbb", "sk-ccc"])
        key = rotator.get_current_key()
        # ... on rate-limit error:
        rotator.rotate_on_error(reason)
        key = rotator.get_current_key()
    """

    keys: list[str] = field(default_factory=list)
    _cooldowns: dict[int, ProfileCooldown] = field(default_factory=dict)
    _current_idx: int = 0

    def get_current_key(self) -> str | None:
        """Return the current active key, skipping keys in cooldown.

        Returns None if all keys are in cooldown.
        """
        if not self.keys:
            return None

        # Clear expired cooldowns first.
        for cd in self._cooldowns.values():
            cd.clear_if_expired()

        # Try each key starting from current index.
        for _ in range(len(self.keys)):
            cd = self._cooldowns.get(self._current_idx)
            if cd is None or not cd.is_in_cooldown:
                return self.keys[self._current_idx]
            self._current_idx = (self._current_idx + 1) % len(self.keys)

        return None  # all keys in cooldown

    def rotate_on_error(self, reason: FailoverReason = FailoverReason.RATE_LIMIT) -> str | None:
        """Record failure on the current key and rotate to the next available.

        Returns the next available key, or None if all are in cooldown.
        """
        if not self.keys:
            return None

        # Mark failure on current key.
        if self._current_idx not in self._cooldowns:
            self._cooldowns[self._current_idx] = ProfileCooldown()
        self._cooldowns[self._current_idx].mark_failure(reason)

        # Rotate.
        self._current_idx = (self._current_idx + 1) % len(self.keys)
        return self.get_current_key()

    def mark_success(self) -> None:
        """Record success on the current key — clears its cooldown."""
        if self._current_idx in self._cooldowns:
            self._cooldowns[self._current_idx].mark_success()


# ---------------------------------------------------------------------------
# Overload pacing (brief backoff between rotation attempts during overload)
# ---------------------------------------------------------------------------

_OVERLOAD_INITIAL_MS = 250
_OVERLOAD_FACTOR = 2
_OVERLOAD_MAX_MS = 1500
_OVERLOAD_JITTER = 0.2


def _overload_delay_seconds(attempt: int) -> float:
    """Compute the overload pacing delay for a given attempt number (0-based).

    Uses exponential backoff: 250ms -> 500ms -> 1000ms -> 1500ms (capped),
    with a small jitter factor to avoid thundering-herd.
    """
    import random

    raw_ms = _OVERLOAD_INITIAL_MS * (_OVERLOAD_FACTOR ** attempt)
    capped_ms = min(raw_ms, _OVERLOAD_MAX_MS)
    jitter = 1.0 + (random.random() * 2 - 1) * _OVERLOAD_JITTER
    return (capped_ms * jitter) / 1000.0


# ---------------------------------------------------------------------------
# Retry iteration guard
# ---------------------------------------------------------------------------

MAX_RETRY_ITERATIONS = 32


# ---------------------------------------------------------------------------
# FailoverManager
# ---------------------------------------------------------------------------

_DEFAULT_STATE_PATH = os.path.expanduser("~/.openclaw-py/failover_state.json")
_DEFAULT_PROBE_INTERVAL = 30  # seconds between probe attempts


@dataclass
class FailoverManager:
    """Manages auth profile rotation, model failover, and retry guards."""

    profiles: list[str] = field(default_factory=lambda: ["default"])
    fallback_models: list[str] = field(default_factory=list)
    cooldowns: dict[str, ProfileCooldown] = field(default_factory=dict)
    state_path: str = _DEFAULT_STATE_PATH
    _current_profile_idx: int = 0
    _current_model_idx: int = -1  # -1 = primary model
    _session_pinned_profile: str | None = None
    _retry_count: int = 0
    _overload_failover_attempts: int = 0
    _last_probe_time: float = 0.0
    _probe_interval: int = _DEFAULT_PROBE_INTERVAL

    def __post_init__(self) -> None:
        self.load_state()

    @property
    def current_profile(self) -> str:
        if self._session_pinned_profile:
            return self._session_pinned_profile
        if not self.profiles:
            return "default"
        return self.profiles[self._current_profile_idx % len(self.profiles)]

    @property
    def retry_count(self) -> int:
        return self._retry_count

    @property
    def retries_exhausted(self) -> bool:
        return self._retry_count >= MAX_RETRY_ITERATIONS

    def pin_profile(self, profile: str) -> None:
        """Pin a profile for the session (session stickiness)."""
        self._session_pinned_profile = profile

    def unpin_profile(self) -> None:
        self._session_pinned_profile = None

    def get_cooldown(self, profile: str) -> ProfileCooldown:
        if profile not in self.cooldowns:
            self.cooldowns[profile] = ProfileCooldown()
        return self.cooldowns[profile]

    def clear_expired_cooldowns(self) -> bool:
        """Reset error counters for all profiles whose cooldown window has
        passed (circuit-breaker half-open -> closed).

        Returns True if any profile was cleared.
        """
        cleared = False
        for cd in self.cooldowns.values():
            if cd.clear_if_expired():
                cleared = True
        return cleared

    def advance_profile(self) -> str | None:
        """Rotate to the next non-cooled-down profile.

        Returns the new profile name, or None if all are in cooldown.
        """
        self.unpin_profile()
        if not self.profiles:
            return None

        # Before rotating, clear any expired cooldowns so profiles get a
        # fair retry window.
        self.clear_expired_cooldowns()

        for _ in range(len(self.profiles)):
            self._current_profile_idx = (self._current_profile_idx + 1) % len(self.profiles)
            profile = self.profiles[self._current_profile_idx]
            if not self.get_cooldown(profile).is_in_cooldown:
                self.pin_profile(profile)
                return profile

        return None  # all profiles in cooldown

    def advance_model(self) -> str | None:
        """Advance to the next fallback model.

        Returns the new model name, or None if exhausted.
        """
        if not self.fallback_models:
            return None
        self._current_model_idx += 1
        if self._current_model_idx >= len(self.fallback_models):
            return None
        return self.fallback_models[self._current_model_idx]

    def _maybe_pace_overload(self, reason: FailoverReason) -> None:
        """If the error is an overload, sleep briefly to avoid tight retry
        bursts. The delay increases with consecutive overload attempts."""
        if reason != FailoverReason.OVERLOADED:
            self._overload_failover_attempts = 0
            return
        delay = _overload_delay_seconds(self._overload_failover_attempts)
        self._overload_failover_attempts += 1
        time.sleep(delay)

    def handle_error(self, error: str | Exception) -> tuple[FailoverReason, str | None]:
        """Handle an error: classify, update cooldown, attempt failover.

        Returns ``(reason, next_model_or_none)``.

        * ``next_model_or_none`` is a fallback model name when profile rotation
          is exhausted, or ``None`` (same model, different profile — or fully
          exhausted).
        * Raises ``RuntimeError`` if the retry iteration guard is exceeded.
        """
        self._retry_count += 1
        if self.retries_exhausted:
            reason = classify_error(error)
            raise RuntimeError(
                f"Exceeded retry limit after {self._retry_count} attempts "
                f"(last reason: {reason.value})"
            )

        reason = classify_error(error)
        profile = self.current_profile

        # Record the failure on the profile's cooldown tracker.
        # Note: mark_failure() internally skips counter increments for TIMEOUT.
        self.get_cooldown(profile).mark_failure(reason)

        if not should_failover(reason):
            self.save_state()
            return reason, None

        # Overload pacing: brief sleep before trying the next profile/model.
        self._maybe_pace_overload(reason)

        # Try rotating profiles first.
        next_profile = self.advance_profile()
        if next_profile:
            self.save_state()
            return reason, None  # same model, different profile

        # All profiles exhausted, try fallback model.
        next_model = self.advance_model()
        if next_model:
            # Reset profile rotation for new model.
            self._current_profile_idx = 0
            self.unpin_profile()
            self.save_state()
            return reason, next_model

        self.save_state()
        return reason, None  # fully exhausted

    def mark_success(self) -> None:
        """Record a successful request — resets cooldown and retry counter for
        the current profile."""
        profile = self.current_profile
        self.get_cooldown(profile).mark_success()
        self.pin_profile(profile)
        self._retry_count = 0
        self._overload_failover_attempts = 0
        self.save_state()

    # -------------------------------------------------------------------
    # State persistence
    # -------------------------------------------------------------------

    def save_state(self) -> None:
        """Persist failover state to disk (JSON) with file locking."""
        state = {
            "current_profile_idx": self._current_profile_idx,
            "current_model_idx": self._current_model_idx,
            "retry_count": self._retry_count,
            "last_probe_time": self._last_probe_time,
            "cooldowns": {
                name: {
                    "error_count": cd.error_count,
                    "billing_error_count": cd.billing_error_count,
                    "cooldown_until": cd.cooldown_until,
                    "last_success": cd.last_success,
                    "last_failure_at": cd.last_failure_at,
                }
                for name, cd in self.cooldowns.items()
            },
        }
        path = Path(self.state_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        try:
            fd = os.open(str(tmp), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX)
                os.write(fd, json.dumps(state, indent=2).encode())
                os.fsync(fd)
            finally:
                fcntl.flock(fd, fcntl.LOCK_UN)
                os.close(fd)
            os.replace(str(tmp), str(path))
        except OSError:
            # Best-effort: don't crash if state dir is unwritable.
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass

    def load_state(self) -> None:
        """Load persisted failover state from disk if it exists."""
        path = Path(self.state_path)
        if not path.is_file():
            return
        try:
            fd = os.open(str(path), os.O_RDONLY)
            try:
                fcntl.flock(fd, fcntl.LOCK_SH)
                raw = os.read(fd, 1_000_000)
            finally:
                fcntl.flock(fd, fcntl.LOCK_UN)
                os.close(fd)
            state = json.loads(raw)
        except (OSError, json.JSONDecodeError):
            return

        self._current_profile_idx = state.get("current_profile_idx", 0)
        self._current_model_idx = state.get("current_model_idx", -1)
        self._retry_count = state.get("retry_count", 0)
        self._last_probe_time = state.get("last_probe_time", 0.0)

        for name, cd_data in state.get("cooldowns", {}).items():
            cd = ProfileCooldown(
                error_count=cd_data.get("error_count", 0),
                billing_error_count=cd_data.get("billing_error_count", 0),
                cooldown_until=cd_data.get("cooldown_until", 0.0),
                last_success=cd_data.get("last_success", 0.0),
                last_failure_at=cd_data.get("last_failure_at", 0.0),
            )
            self.cooldowns[name] = cd

    # -------------------------------------------------------------------
    # Probe mechanism
    # -------------------------------------------------------------------

    def should_probe_primary(self) -> bool:
        """Check whether we should probe the primary model.

        Returns True when:
        - We are currently on a fallback model (not primary).
        - Enough time has passed since the last probe (``_probe_interval``).
        - Either the probe interval has elapsed, or the primary's cooldown
          will expire within 2 minutes.
        """
        # Only probe when we're on a fallback.
        if self._current_model_idx < 0:
            return False

        now = time.time()
        elapsed = now - self._last_probe_time
        if elapsed < self._probe_interval:
            return False

        # Check if any primary profile cooldown is near expiry (within 2 min).
        for profile in self.profiles:
            cd = self.cooldowns.get(profile)
            if cd is None or not cd.is_in_cooldown:
                # A primary profile is already available — probe.
                return True
            remaining = cd.cooldown_until - now
            if remaining <= 120:
                return True

        # Fallback: allow periodic probes even if cooldowns are long.
        return elapsed >= self._probe_interval

    def probe_primary(self, success: bool) -> None:
        """Record the result of a primary-model probe.

        If *success* is True, reset to primary model and clear cooldowns.
        If False, stay on fallback and record the probe time.
        """
        now = time.time()
        self._last_probe_time = now

        if success:
            # Reset to primary model.
            self._current_model_idx = -1
            self._current_profile_idx = 0
            self.unpin_profile()
            # Clear all primary profile cooldowns.
            for profile in self.profiles:
                cd = self.cooldowns.get(profile)
                if cd is not None:
                    cd.mark_success()
            self._retry_count = 0
            self._overload_failover_attempts = 0

        self.save_state()
