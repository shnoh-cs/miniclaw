"""Error classification and model failover logic.

FailoverManager orchestrates profile rotation, model fallback, and state
persistence.  Error classification and cooldown tracking are split into
``error_classify`` and ``cooldown`` submodules.
"""

from __future__ import annotations

import fcntl
import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

from openclaw.agent.types import FailoverReason

# Re-export public API for backward compatibility
from openclaw.model.error_classify import (  # noqa: F401
    classify_error,
    should_failover,
)
from openclaw.model.cooldown import (  # noqa: F401
    ApiKeyRotator,
    ProfileCooldown,
    overload_delay_seconds,
)

log = logging.getLogger("openclaw.failover")

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
        self._session_pinned_profile = profile

    def unpin_profile(self) -> None:
        self._session_pinned_profile = None

    def get_cooldown(self, profile: str) -> ProfileCooldown:
        if profile not in self.cooldowns:
            self.cooldowns[profile] = ProfileCooldown()
        return self.cooldowns[profile]

    def clear_expired_cooldowns(self) -> bool:
        """Reset error counters for profiles whose cooldown window has passed."""
        cleared = False
        for cd in self.cooldowns.values():
            if cd.clear_if_expired():
                cleared = True
        return cleared

    def advance_profile(self) -> str | None:
        """Rotate to the next non-cooled-down profile."""
        self.unpin_profile()
        if not self.profiles:
            return None

        self.clear_expired_cooldowns()

        for _ in range(len(self.profiles)):
            self._current_profile_idx = (self._current_profile_idx + 1) % len(self.profiles)
            profile = self.profiles[self._current_profile_idx]
            if not self.get_cooldown(profile).is_in_cooldown:
                self.pin_profile(profile)
                return profile

        return None

    def advance_model(self) -> str | None:
        """Advance to the next fallback model."""
        if not self.fallback_models:
            return None
        self._current_model_idx += 1
        if self._current_model_idx >= len(self.fallback_models):
            return None
        return self.fallback_models[self._current_model_idx]

    def _maybe_pace_overload(self, reason: FailoverReason) -> None:
        """Brief sleep during overload to avoid tight retry bursts."""
        if reason != FailoverReason.OVERLOADED:
            self._overload_failover_attempts = 0
            return
        delay = overload_delay_seconds(self._overload_failover_attempts)
        self._overload_failover_attempts += 1
        time.sleep(delay)

    def handle_error(self, error: str | Exception) -> tuple[FailoverReason, str | None]:
        """Handle an error: classify, update cooldown, attempt failover.

        Returns ``(reason, next_model_or_none)``.
        Raises ``RuntimeError`` if the retry iteration guard is exceeded.
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

        self.get_cooldown(profile).mark_failure(reason)

        if not should_failover(reason):
            self.save_state()
            return reason, None

        self._maybe_pace_overload(reason)

        next_profile = self.advance_profile()
        if next_profile:
            self.save_state()
            return reason, None

        next_model = self.advance_model()
        if next_model:
            self._current_profile_idx = 0
            self.unpin_profile()
            self.save_state()
            return reason, next_model

        self.save_state()
        return reason, None

    def mark_success(self) -> None:
        """Record a successful request — resets cooldown and retry counter."""
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
            log.debug("Failed to save failover state", exc_info=True)
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
            log.debug("Failed to load failover state", exc_info=True)
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
        """Check whether we should probe the primary model."""
        if self._current_model_idx < 0:
            return False

        now = time.time()
        elapsed = now - self._last_probe_time
        if elapsed < self._probe_interval:
            return False

        for profile in self.profiles:
            cd = self.cooldowns.get(profile)
            if cd is None or not cd.is_in_cooldown:
                return True
            remaining = cd.cooldown_until - now
            if remaining <= 120:
                return True

        return elapsed >= self._probe_interval

    def probe_primary(self, success: bool) -> None:
        """Record the result of a primary-model probe."""
        now = time.time()
        self._last_probe_time = now

        if success:
            self._current_model_idx = -1
            self._current_profile_idx = 0
            self.unpin_profile()
            for profile in self.profiles:
                cd = self.cooldowns.get(profile)
                if cd is not None:
                    cd.mark_success()
            self._retry_count = 0
            self._overload_failover_attempts = 0

        self.save_state()
