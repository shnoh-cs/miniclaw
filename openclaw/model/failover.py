"""Error classification and model failover logic."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

from openclaw.agent.types import FailoverReason


# ---------------------------------------------------------------------------
# Error classification
# ---------------------------------------------------------------------------

_AUTH_PATTERNS = re.compile(
    r"(invalid.?api.?key|unauthorized|authentication|forbidden|401|403)",
    re.IGNORECASE,
)
_BILLING_PATTERNS = re.compile(
    r"(insufficient.?credits|credit.?balance|billing|payment.?required|quota.?exceeded)",
    re.IGNORECASE,
)
_RATE_LIMIT_PATTERNS = re.compile(
    r"(rate.?limit|too.?many.?requests|429|throttl)",
    re.IGNORECASE,
)
_TIMEOUT_PATTERNS = re.compile(
    r"(timeout|timed?.?out|deadline.?exceeded|504|408)",
    re.IGNORECASE,
)
_CONTEXT_OVERFLOW_PATTERNS = re.compile(
    r"(context.?length|token.?limit|maximum.?context|too.?long|exceed.*(length|limit|tokens))",
    re.IGNORECASE,
)
_MODEL_NOT_FOUND_PATTERNS = re.compile(
    r"(model.?not.?found|does.?not.?exist|unknown.?model|404)",
    re.IGNORECASE,
)


def classify_error(error: str | Exception) -> FailoverReason:
    """Classify an error into a failover reason."""
    text = str(error)

    if _AUTH_PATTERNS.search(text):
        return FailoverReason.AUTH
    if _BILLING_PATTERNS.search(text):
        return FailoverReason.BILLING
    if _RATE_LIMIT_PATTERNS.search(text):
        return FailoverReason.RATE_LIMIT
    if _TIMEOUT_PATTERNS.search(text):
        return FailoverReason.TIMEOUT
    if _CONTEXT_OVERFLOW_PATTERNS.search(text):
        return FailoverReason.CONTEXT_OVERFLOW
    if _MODEL_NOT_FOUND_PATTERNS.search(text):
        return FailoverReason.MODEL_NOT_FOUND

    return FailoverReason.UNKNOWN


def should_failover(reason: FailoverReason) -> bool:
    """Determine if the error warrants failover to another model/profile."""
    return reason in {
        FailoverReason.AUTH,
        FailoverReason.BILLING,
        FailoverReason.RATE_LIMIT,
        FailoverReason.TIMEOUT,
    }


# ---------------------------------------------------------------------------
# Auth profile cooldown tracking
# ---------------------------------------------------------------------------

# Transient: 1m → 5m → 25m → 1h (cap)
_TRANSIENT_BACKOFF_SECONDS = [60, 300, 1500, 3600]
# Billing: 5h → 10h → 24h (cap)
_BILLING_BACKOFF_SECONDS = [18000, 36000, 86400]


@dataclass
class ProfileCooldown:
    """Tracks cooldown state for an auth profile."""

    error_count: int = 0
    billing_error_count: int = 0
    cooldown_until: float = 0.0
    last_success: float = 0.0

    @property
    def is_in_cooldown(self) -> bool:
        return time.time() < self.cooldown_until

    def mark_failure(self, reason: FailoverReason) -> None:
        now = time.time()
        if reason == FailoverReason.BILLING:
            idx = min(self.billing_error_count, len(_BILLING_BACKOFF_SECONDS) - 1)
            backoff = _BILLING_BACKOFF_SECONDS[idx]
            self.billing_error_count += 1
        else:
            idx = min(self.error_count, len(_TRANSIENT_BACKOFF_SECONDS) - 1)
            backoff = _TRANSIENT_BACKOFF_SECONDS[idx]
            self.error_count += 1
        self.cooldown_until = now + backoff

    def mark_success(self) -> None:
        self.error_count = 0
        self.billing_error_count = 0
        self.cooldown_until = 0.0
        self.last_success = time.time()

    def reset_if_stale(self, stale_hours: int = 24) -> None:
        """Reset counters if last success was recent enough."""
        if self.last_success and (time.time() - self.last_success) < stale_hours * 3600:
            self.error_count = 0
            self.billing_error_count = 0


@dataclass
class FailoverManager:
    """Manages auth profile rotation and model failover."""

    profiles: list[str] = field(default_factory=lambda: ["default"])
    fallback_models: list[str] = field(default_factory=list)
    cooldowns: dict[str, ProfileCooldown] = field(default_factory=dict)
    _current_profile_idx: int = 0
    _current_model_idx: int = -1  # -1 = primary model
    _session_pinned_profile: str | None = None

    @property
    def current_profile(self) -> str:
        if self._session_pinned_profile:
            return self._session_pinned_profile
        if not self.profiles:
            return "default"
        return self.profiles[self._current_profile_idx % len(self.profiles)]

    def pin_profile(self, profile: str) -> None:
        """Pin a profile for the session (session stickiness)."""
        self._session_pinned_profile = profile

    def unpin_profile(self) -> None:
        self._session_pinned_profile = None

    def get_cooldown(self, profile: str) -> ProfileCooldown:
        if profile not in self.cooldowns:
            self.cooldowns[profile] = ProfileCooldown()
        return self.cooldowns[profile]

    def advance_profile(self) -> str | None:
        """Rotate to the next non-cooled-down profile.

        Returns the new profile name, or None if all are in cooldown.
        """
        self.unpin_profile()
        if not self.profiles:
            return None

        for i in range(len(self.profiles)):
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

    def handle_error(self, error: str | Exception) -> tuple[FailoverReason, str | None]:
        """Handle an error: classify, update cooldown, attempt failover.

        Returns (reason, next_model_or_none).
        """
        reason = classify_error(error)
        profile = self.current_profile
        self.get_cooldown(profile).mark_failure(reason)

        if not should_failover(reason):
            return reason, None

        # Try rotating profiles first
        next_profile = self.advance_profile()
        if next_profile:
            return reason, None  # same model, different profile

        # All profiles exhausted, try fallback model
        next_model = self.advance_model()
        if next_model:
            # Reset profile rotation for new model
            self._current_profile_idx = 0
            self.unpin_profile()
            return reason, next_model

        return reason, None  # fully exhausted

    def mark_success(self) -> None:
        profile = self.current_profile
        self.get_cooldown(profile).mark_success()
        self.pin_profile(profile)
