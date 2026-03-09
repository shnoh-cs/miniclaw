"""Auth profile cooldown tracking, API key rotation, and overload pacing."""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field

from openclaw.agent.types import FailoverReason

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

        Timeout errors trigger rotation but do NOT increment error counters
        or set cooldown windows.
        """
        now = time.time()
        self.last_failure_at = now

        if reason == FailoverReason.TIMEOUT:
            return

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
        """Clear cooldown and reset counters if the window has passed."""
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
    """Manages multiple API keys with round-robin rotation and per-key cooldowns."""

    keys: list[str] = field(default_factory=list)
    _cooldowns: dict[int, ProfileCooldown] = field(default_factory=dict)
    _current_idx: int = 0

    def get_current_key(self) -> str | None:
        """Return the current active key, skipping keys in cooldown."""
        if not self.keys:
            return None

        for cd in self._cooldowns.values():
            cd.clear_if_expired()

        for _ in range(len(self.keys)):
            cd = self._cooldowns.get(self._current_idx)
            if cd is None or not cd.is_in_cooldown:
                return self.keys[self._current_idx]
            self._current_idx = (self._current_idx + 1) % len(self.keys)

        return None

    def rotate_on_error(self, reason: FailoverReason = FailoverReason.RATE_LIMIT) -> str | None:
        """Record failure on the current key and rotate to the next available."""
        if not self.keys:
            return None

        if self._current_idx not in self._cooldowns:
            self._cooldowns[self._current_idx] = ProfileCooldown()
        self._cooldowns[self._current_idx].mark_failure(reason)

        self._current_idx = (self._current_idx + 1) % len(self.keys)
        return self.get_current_key()

    def mark_success(self) -> None:
        """Record success on the current key — clears its cooldown."""
        if self._current_idx in self._cooldowns:
            self._cooldowns[self._current_idx].mark_success()


# ---------------------------------------------------------------------------
# Overload pacing
# ---------------------------------------------------------------------------

_OVERLOAD_INITIAL_MS = 250
_OVERLOAD_FACTOR = 2
_OVERLOAD_MAX_MS = 1500
_OVERLOAD_JITTER = 0.2

_RATE_LIMIT_INITIAL_MS = 1000
_RATE_LIMIT_FACTOR = 2
_RATE_LIMIT_MAX_MS = 30_000
_RATE_LIMIT_JITTER = 0.2


def overload_delay_seconds(attempt: int) -> float:
    """Compute the overload pacing delay for a given attempt number (0-based).

    Uses exponential backoff: 250ms -> 500ms -> 1000ms -> 1500ms (capped),
    with a small jitter factor.
    """
    raw_ms = _OVERLOAD_INITIAL_MS * (_OVERLOAD_FACTOR ** attempt)
    capped_ms = min(raw_ms, _OVERLOAD_MAX_MS)
    jitter = 1.0 + (random.random() * 2 - 1) * _OVERLOAD_JITTER
    return (capped_ms * jitter) / 1000.0


def rate_limit_delay_seconds(attempt: int) -> float:
    """Compute rate-limit backoff delay (0-based attempt).

    Uses exponential backoff: 1s -> 2s -> 4s -> 8s -> 16s -> 30s (capped),
    with jitter.
    """
    raw_ms = _RATE_LIMIT_INITIAL_MS * (_RATE_LIMIT_FACTOR ** attempt)
    capped_ms = min(raw_ms, _RATE_LIMIT_MAX_MS)
    jitter = 1.0 + (random.random() * 2 - 1) * _RATE_LIMIT_JITTER
    return (capped_ms * jitter) / 1000.0
