"""Tool loop detection: identifies stuck tool call patterns.

4 detector types:
1. generic_repeat: same tool+params repeated (warning-only)
2. known_poll_no_progress: polling with identical results
3. ping_pong: alternating A->B->A->B with no-progress evidence
4. global_circuit_breaker: per-tool no-progress streak
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any


def _hash_value(data: str) -> str:
    """Short SHA-256 hex digest."""
    return hashlib.sha256(data.encode()).hexdigest()[:16]


def _hash_call(name: str, args: dict[str, Any]) -> str:
    """Deterministic hash of a tool call (name:digest)."""
    stable = json.dumps({"name": name, "args": args}, sort_keys=True)
    return f"{name}:{_hash_value(stable)}"


def _hash_result(result: str) -> str:
    """Hash a tool result string."""
    return _hash_value(result) if result else ""


def _is_known_poll_tool(name: str, args: dict[str, Any]) -> bool:
    """Identify polling tools (process with action=poll|log, command_status)."""
    if name == "command_status":
        return True
    if name != "process":
        return False
    action = args.get("action", "")
    return action in ("poll", "log")


@dataclass
class _HistoryRecord:
    """Single entry in the tool call history."""

    tool_name: str
    args_hash: str
    result_hash: str | None = None


@dataclass
class LoopDetectionResult:
    """Result from loop detection analysis."""

    stuck: bool = False
    level: str | None = None  # "warning" | "critical"
    detector: str | None = None  # detector kind
    count: int = 0
    message: str = ""
    paired_tool_name: str | None = None
    warning_key: str | None = None


# Warning deduplication constants
_LOOP_WARNING_BUCKET_SIZE = 10
_MAX_LOOP_WARNING_KEYS = 256


def _as_positive_int(value: int, fallback: int) -> int:
    """Ensure value is a positive integer, else return fallback."""
    if not isinstance(value, int) or value <= 0:
        return fallback
    return value


def _canonical_pair_key(sig_a: str, sig_b: str) -> str:
    """Stable key for a pair of signatures regardless of order."""
    return "|".join(sorted([sig_a, sig_b]))


@dataclass
class ToolLoopDetector:
    """Detects stuck tool call patterns.

    Two-phase recording:
    - record_call(name, args) before execution
    - record_outcome(result) after execution
    Legacy record() still works for backward compatibility.

    Warning deduplication: bucket-based, one warning per 10-count bucket per key.
    """

    history: list[_HistoryRecord] = field(default_factory=list)
    window_size: int = 30
    warning_threshold: int = 10
    critical_threshold: int = 20
    breaker_threshold: int = 30

    _warning_buckets: dict[str, int] = field(default_factory=dict)
    _pending_name: str | None = field(default=None, repr=False)
    _pending_args: dict[str, Any] | None = field(default=None, repr=False)
    _pending_args_hash: str | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        self.warning_threshold = _as_positive_int(self.warning_threshold, 10)
        self.critical_threshold = _as_positive_int(self.critical_threshold, 20)
        self.breaker_threshold = _as_positive_int(self.breaker_threshold, 30)
        if self.critical_threshold <= self.warning_threshold:
            self.critical_threshold = self.warning_threshold + 1
        if self.breaker_threshold <= self.critical_threshold:
            self.breaker_threshold = self.critical_threshold + 1

    # ------------------------------------------------------------------
    # Two-phase recording
    # ------------------------------------------------------------------

    def record_call(self, name: str, args: dict[str, Any]) -> None:
        """Phase 1: record a tool call before execution."""
        args_hash = _hash_call(name, args)
        self.history.append(_HistoryRecord(tool_name=name, args_hash=args_hash))
        self._pending_name = name
        self._pending_args = args
        self._pending_args_hash = args_hash
        self._trim_history()

    def record_outcome(self, result: str = "") -> None:
        """Phase 2: attach a result hash to the most recent pending call."""
        if self._pending_args_hash is None:
            return
        rh = _hash_result(result)
        for i in range(len(self.history) - 1, -1, -1):
            rec = self.history[i]
            if rec.args_hash == self._pending_args_hash and rec.result_hash is None:
                rec.result_hash = rh
                break
        self._pending_name = None
        self._pending_args = None
        self._pending_args_hash = None

    # ------------------------------------------------------------------
    # Legacy single-step recording
    # ------------------------------------------------------------------

    def record(self, name: str, args: dict[str, Any], result: str = "") -> str | None:
        """Record a tool call + result in one step. Returns warning/error or None."""
        self.record_call(name, args)
        self.record_outcome(result)
        detection = self.detect(name, args)
        if detection.stuck:
            if self._should_emit_warning(detection.warning_key or "", detection.count):
                return detection.message
        return None

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------

    def detect(self, name: str, args: dict[str, Any]) -> LoopDetectionResult:
        """Run all detectors against current history."""
        current_hash = _hash_call(name, args)
        is_poll = _is_known_poll_tool(name, args)

        # 4. Global circuit breaker
        no_progress = self._get_no_progress_streak(name, current_hash)
        if no_progress["count"] >= self.breaker_threshold:
            return LoopDetectionResult(
                stuck=True,
                level="critical",
                detector="global_circuit_breaker",
                count=no_progress["count"],
                message=(
                    f"CRITICAL: {name} has repeated identical no-progress outcomes "
                    f"{no_progress['count']} times. Session execution blocked by "
                    f"global circuit breaker to prevent runaway loops."
                ),
                warning_key=f"global:{name}:{current_hash}:{no_progress.get('latest_hash', 'none')}",
            )

        # 2. Known poll no-progress
        if is_poll:
            if no_progress["count"] >= self.critical_threshold:
                return LoopDetectionResult(
                    stuck=True,
                    level="critical",
                    detector="known_poll_no_progress",
                    count=no_progress["count"],
                    message=(
                        f"CRITICAL: Called {name} with identical arguments and no progress "
                        f"{no_progress['count']} times. This appears to be a stuck polling "
                        f"loop. Session execution blocked to prevent resource waste."
                    ),
                    warning_key=f"poll:{name}:{current_hash}:{no_progress.get('latest_hash', 'none')}",
                )
            if no_progress["count"] >= self.warning_threshold:
                return LoopDetectionResult(
                    stuck=True,
                    level="warning",
                    detector="known_poll_no_progress",
                    count=no_progress["count"],
                    message=(
                        f"WARNING: You have called {name} {no_progress['count']} times with "
                        f"identical arguments and no progress. Stop polling and either "
                        f"(1) increase wait time between checks, or (2) report the task "
                        f"as failed if the process is stuck."
                    ),
                    warning_key=f"poll:{name}:{current_hash}:{no_progress.get('latest_hash', 'none')}",
                )

        # 3. Ping-pong
        ping_pong = self._get_ping_pong_streak(current_hash)
        pp_warning_key = (
            f"pingpong:{_canonical_pair_key(current_hash, ping_pong['paired_signature'])}"
            if ping_pong.get("paired_signature")
            else f"pingpong:{name}:{current_hash}"
        )

        if (
            ping_pong["count"] >= self.critical_threshold
            and ping_pong["no_progress_evidence"]
        ):
            return LoopDetectionResult(
                stuck=True,
                level="critical",
                detector="ping_pong",
                count=ping_pong["count"],
                message=(
                    f"CRITICAL: You are alternating between repeated tool-call patterns "
                    f"({ping_pong['count']} consecutive calls) with no progress. This "
                    f"appears to be a stuck ping-pong loop. Session execution blocked "
                    f"to prevent resource waste."
                ),
                paired_tool_name=ping_pong.get("paired_tool_name"),
                warning_key=pp_warning_key,
            )

        if ping_pong["count"] >= self.warning_threshold:
            return LoopDetectionResult(
                stuck=True,
                level="warning",
                detector="ping_pong",
                count=ping_pong["count"],
                message=(
                    f"WARNING: You are alternating between repeated tool-call patterns "
                    f"({ping_pong['count']} consecutive calls). This looks like a "
                    f"ping-pong loop; stop retrying and report the task as failed."
                ),
                paired_tool_name=ping_pong.get("paired_tool_name"),
                warning_key=pp_warning_key,
            )

        # 1. Generic repeat
        if not is_poll:
            recent_count = sum(
                1 for rec in self.history
                if rec.tool_name == name and rec.args_hash == current_hash
            )
            if recent_count >= self.warning_threshold:
                return LoopDetectionResult(
                    stuck=True,
                    level="warning",
                    detector="generic_repeat",
                    count=recent_count,
                    message=(
                        f"WARNING: You have called {name} {recent_count} times with "
                        f"identical arguments. If this is not making progress, stop "
                        f"retrying and report the task as failed."
                    ),
                    warning_key=f"generic:{name}:{current_hash}",
                )

        return LoopDetectionResult(stuck=False)

    # ------------------------------------------------------------------
    # Warning deduplication
    # ------------------------------------------------------------------

    def _should_emit_warning(self, warning_key: str, count: int) -> bool:
        if not warning_key:
            return True
        bucket = count // _LOOP_WARNING_BUCKET_SIZE
        last_bucket = self._warning_buckets.get(warning_key, -1)
        if bucket <= last_bucket:
            return False
        self._warning_buckets[warning_key] = bucket
        if len(self._warning_buckets) > _MAX_LOOP_WARNING_KEYS:
            oldest = next(iter(self._warning_buckets))
            del self._warning_buckets[oldest]
        return True

    def should_emit_warning(self, warning_key: str, count: int) -> bool:
        """Public interface for warning deduplication."""
        return self._should_emit_warning(warning_key, count)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _trim_history(self) -> None:
        if len(self.history) > self.window_size:
            self.history = self.history[-self.window_size:]

    def _get_no_progress_streak(
        self, tool_name: str, args_hash: str
    ) -> dict[str, Any]:
        """Count consecutive identical results for the given tool from the back."""
        streak = 0
        latest_hash: str | None = None

        for i in range(len(self.history) - 1, -1, -1):
            rec = self.history[i]
            if rec.tool_name != tool_name or rec.args_hash != args_hash:
                continue
            if rec.result_hash is None or rec.result_hash == "":
                continue
            if latest_hash is None:
                latest_hash = rec.result_hash
                streak = 1
                continue
            if rec.result_hash != latest_hash:
                break
            streak += 1

        return {"count": streak, "latest_hash": latest_hash or "none"}

    def _get_ping_pong_streak(self, current_signature: str) -> dict[str, Any]:
        """Dynamic streak measurement: walk backward counting alternating pattern."""
        if not self.history:
            return {"count": 0, "no_progress_evidence": False}

        last = self.history[-1]

        other_signature: str | None = None
        other_tool_name: str | None = None
        for i in range(len(self.history) - 2, -1, -1):
            rec = self.history[i]
            if rec.args_hash != last.args_hash:
                other_signature = rec.args_hash
                other_tool_name = rec.tool_name
                break

        if other_signature is None or other_tool_name is None:
            return {"count": 0, "no_progress_evidence": False}

        alternating_count = 0
        for i in range(len(self.history) - 1, -1, -1):
            rec = self.history[i]
            expected = last.args_hash if alternating_count % 2 == 0 else other_signature
            if rec.args_hash != expected:
                break
            alternating_count += 1

        if alternating_count < 2:
            return {"count": 0, "no_progress_evidence": False}

        if current_signature != other_signature:
            return {"count": 0, "no_progress_evidence": False}

        tail_start = max(0, len(self.history) - alternating_count)
        first_hash_a: str | None = None
        first_hash_b: str | None = None
        no_progress_evidence = True

        for i in range(tail_start, len(self.history)):
            rec = self.history[i]
            if not rec.result_hash:
                no_progress_evidence = False
                break
            if rec.args_hash == last.args_hash:
                if first_hash_a is None:
                    first_hash_a = rec.result_hash
                elif first_hash_a != rec.result_hash:
                    no_progress_evidence = False
                    break
            elif rec.args_hash == other_signature:
                if first_hash_b is None:
                    first_hash_b = rec.result_hash
                elif first_hash_b != rec.result_hash:
                    no_progress_evidence = False
                    break
            else:
                no_progress_evidence = False
                break

        if first_hash_a is None or first_hash_b is None:
            no_progress_evidence = False

        return {
            "count": alternating_count + 1,
            "paired_tool_name": last.tool_name,
            "paired_signature": last.args_hash,
            "no_progress_evidence": no_progress_evidence,
        }

    def reset(self) -> None:
        self.history.clear()
        self._warning_buckets.clear()
        self._pending_name = None
        self._pending_args = None
        self._pending_args_hash = None
