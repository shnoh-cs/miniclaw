"""Tool registry: registration, lookup, schema conversion, and result guarding."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from openclaw.agent.types import ToolDefinition, ToolResult

# Type alias for tool execution functions
ToolExecutor = Callable[[dict[str, Any]], Awaitable[ToolResult]]


@dataclass
class RegisteredTool:
    """A tool registered in the system."""

    definition: ToolDefinition
    executor: ToolExecutor
    group: str = "custom"  # fs, runtime, web, memory, analysis, sessions, custom


class ToolRegistry:
    """Central registry for all available tools."""

    def __init__(self) -> None:
        self._tools: dict[str, RegisteredTool] = {}

    def register(
        self,
        definition: ToolDefinition,
        executor: ToolExecutor,
        group: str = "custom",
    ) -> None:
        self._tools[definition.name] = RegisteredTool(
            definition=definition, executor=executor, group=group
        )

    def unregister(self, name: str) -> None:
        self._tools.pop(name, None)

    def get(self, name: str) -> RegisteredTool | None:
        return self._tools.get(name)

    def get_definitions(self) -> list[ToolDefinition]:
        return [t.definition for t in self._tools.values()]

    def get_names(self) -> list[str]:
        return list(self._tools.keys())

    async def execute(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        tool = self._tools.get(name)
        if not tool:
            return ToolResult(
                tool_use_id="",
                content=f"Error: Unknown tool '{name}'",
                is_error=True,
            )
        try:
            return await tool.executor(arguments)
        except Exception as e:
            return ToolResult(
                tool_use_id="",
                content=f"Error executing tool '{name}': {e}",
                is_error=True,
            )


# ---------------------------------------------------------------------------
# Tool result guard: truncation to prevent context overflow
# ---------------------------------------------------------------------------

HARD_LIMIT_CHARS = 400_000
MIN_KEEP_CHARS = 2_000
_TAIL_CAP = 4_000

TRUNCATION_SUFFIX = (
    "\n\n\u26a0\ufe0f [Content truncated \u2014 original was too large for the model's "
    "context window. The content above is a partial view. If you need more, "
    "request specific sections or use offset/limit parameters to read smaller chunks.]"
)

MIDDLE_OMISSION_MARKER = (
    "\n\n\u26a0\ufe0f [... middle content omitted \u2014 showing head and tail ...]\n\n"
)

# Patterns that indicate the tail of the output is diagnostically important.
_IMPORTANT_TAIL_RE = re.compile(
    r"\b(error|exception|failed|fatal|traceback|panic|stack trace|errno|exit code"
    r"|total|summary|result|complete|finished|done)\b",
    re.IGNORECASE,
)


def _has_important_tail(text: str) -> bool:
    """Detect error/exception/traceback/JSON patterns in the last 2000 chars."""
    tail = text[-2000:]
    if _IMPORTANT_TAIL_RE.search(tail):
        return True
    # JSON closing structure
    if re.search(r"\}\s*$", tail.strip()):
        return True
    return False


def _find_newline_cut(text: str, target: int, lo_ratio: float = 0.8) -> int:
    """Find a clean newline cut point within [lo_ratio*target, target]."""
    lo = int(target * lo_ratio)
    nl = text.rfind("\n", lo, target)
    return nl if nl > lo else target


def truncate_tool_result(content: str, max_chars: int) -> str:
    """Truncate tool result with smart head+tail strategy.

    Uses head+tail only when the tail contains important diagnostic content
    (errors, tracebacks, JSON closing, summaries). Otherwise head-only with
    a clean newline boundary cut. Tail is capped at 4000 chars.
    """
    if len(content) <= max_chars:
        return content

    effective_max = max(max_chars, MIN_KEEP_CHARS)
    if effective_max >= len(content):
        return content

    budget = max(MIN_KEEP_CHARS, effective_max - len(TRUNCATION_SUFFIX))

    # Head+tail mode when tail has important content
    if _has_important_tail(content) and budget > MIN_KEEP_CHARS * 2:
        tail_budget = min(int(budget * 0.3), _TAIL_CAP)
        head_budget = budget - tail_budget - len(MIDDLE_OMISSION_MARKER)

        if head_budget > MIN_KEEP_CHARS:
            head_cut = _find_newline_cut(content, head_budget)
            tail_start = len(content) - tail_budget
            # Try to start tail on a newline boundary
            nl = content.find("\n", tail_start)
            if nl != -1 and nl < tail_start + int(tail_budget * 0.2):
                tail_start = nl + 1
            return (
                content[:head_cut]
                + MIDDLE_OMISSION_MARKER
                + content[tail_start:]
                + TRUNCATION_SUFFIX
            )

    # Default: head-only with clean newline cut
    cut_point = _find_newline_cut(content, budget)
    return content[:cut_point] + TRUNCATION_SUFFIX


# ---------------------------------------------------------------------------
# Tool loop detection
# ---------------------------------------------------------------------------


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
    args_hash: str  # name:digest format
    result_hash: str | None = None  # filled in by record_outcome


# Severity levels as string literals
LoopLevel = str  # "warning" | "critical"

# Detector kinds
LoopDetectorKind = str  # "generic_repeat" | "known_poll_no_progress" | "ping_pong" | "global_circuit_breaker"


@dataclass
class LoopDetectionResult:
    """Result from loop detection analysis."""

    stuck: bool = False
    level: LoopLevel | None = None
    detector: LoopDetectorKind | None = None
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


@dataclass
class ToolLoopDetector:
    """Detects stuck tool call patterns.

    4 detector types:
    1. generic_repeat: same tool+params repeated (warning-only, never blocks)
    2. known_poll_no_progress: polling with identical results (warn@10, critical@20)
    3. ping_pong: alternating A->B->A->B with dynamic streak + no-progress evidence
    4. global_circuit_breaker: per-tool no-progress streak (critical@30)

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

    # Warning deduplication: warning_key -> last emitted bucket number
    _warning_buckets: dict[str, int] = field(default_factory=dict)

    # Pending call awaiting outcome (set by record_call, consumed by record_outcome)
    _pending_name: str | None = field(default=None, repr=False)
    _pending_args: dict[str, Any] | None = field(default=None, repr=False)
    _pending_args_hash: str | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        # Threshold validation: critical > warning, breaker > critical
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
        """Phase 1: record a tool call before execution.

        Creates a history entry without a result_hash. Call record_outcome()
        after execution to attach the result.
        """
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
        # Walk backward to find the matching record without a result yet
        for i in range(len(self.history) - 1, -1, -1):
            rec = self.history[i]
            if rec.args_hash == self._pending_args_hash and rec.result_hash is None:
                rec.result_hash = rh
                break
        self._pending_name = None
        self._pending_args = None
        self._pending_args_hash = None

    # ------------------------------------------------------------------
    # Legacy single-step recording (backward compatible)
    # ------------------------------------------------------------------

    def record(self, name: str, args: dict[str, Any], result: str = "") -> str | None:
        """Record a tool call + result in one step. Returns warning/error or None.

        Backward-compatible entry point. Internally uses two-phase recording
        and then runs detection.
        """
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
        """Run all detectors against current history. Returns detection result."""
        current_hash = _hash_call(name, args)
        is_poll = _is_known_poll_tool(name, args)

        # 4. Global circuit breaker: per-tool no-progress streak
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

        # 3. Ping-pong: dynamic streak + no-progress evidence
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

        # 1. Generic repeat: warning-only, never blocks
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
        """Bucket-based dedup: emit once per 10-count bucket per unique key."""
        if not warning_key:
            return True
        bucket = count // _LOOP_WARNING_BUCKET_SIZE
        last_bucket = self._warning_buckets.get(warning_key, -1)
        if bucket <= last_bucket:
            return False
        self._warning_buckets[warning_key] = bucket
        # Evict oldest keys if map grows too large
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
            self.history = self.history[-self.window_size :]

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

        # Find the "other" signature (first different one walking backward)
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

        # Count alternating tail length
        alternating_count = 0
        for i in range(len(self.history) - 1, -1, -1):
            rec = self.history[i]
            expected = last.args_hash if alternating_count % 2 == 0 else other_signature
            if rec.args_hash != expected:
                break
            alternating_count += 1

        if alternating_count < 2:
            return {"count": 0, "no_progress_evidence": False}

        # Verify current call matches expected position
        if current_signature != other_signature:
            return {"count": 0, "no_progress_evidence": False}

        # Check no-progress evidence: both sides produce identical results
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

        # Need stable outcomes on both sides
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


def _canonical_pair_key(sig_a: str, sig_b: str) -> str:
    """Stable key for a pair of signatures regardless of order."""
    return "|".join(sorted([sig_a, sig_b]))
