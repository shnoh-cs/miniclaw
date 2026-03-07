"""Tool registry: registration, lookup, schema conversion, and result guarding."""

from __future__ import annotations

import hashlib
import json
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


def truncate_tool_result(content: str, max_chars: int) -> str:
    """Truncate tool result with head+tail strategy.

    Preserves the beginning (likely contains the answer) and end
    (likely contains error messages or summaries).
    """
    if len(content) <= max_chars:
        return content

    effective_max = max(max_chars, MIN_KEEP_CHARS)
    if effective_max >= len(content):
        return content

    head_size = int(effective_max * 0.7)
    tail_size = effective_max - head_size
    marker = f"\n\n⚠️ [Content truncated: {len(content)} chars → {effective_max} chars]\n\n"

    return content[:head_size] + marker + content[-tail_size:]


# ---------------------------------------------------------------------------
# Tool loop detection
# ---------------------------------------------------------------------------


def _hash_call(name: str, args: dict[str, Any]) -> str:
    """Deterministic hash of a tool call."""
    stable = json.dumps({"name": name, "args": args}, sort_keys=True)
    return hashlib.sha256(stable.encode()).hexdigest()[:16]


@dataclass
class ToolLoopDetector:
    """Detects stuck tool call patterns.

    4 detector types:
    1. generic_repeat: same tool+params repeated N times
    2. known_poll_no_progress: polling with identical results
    3. ping_pong: alternating A→B→A→B pattern
    4. global_circuit_breaker: total call count threshold
    """

    history: list[str] = field(default_factory=list)  # hashes
    result_hashes: list[str] = field(default_factory=list)
    window_size: int = 30
    warning_threshold: int = 10
    critical_threshold: int = 20
    breaker_threshold: int = 30
    total_calls: int = 0

    def record(self, name: str, args: dict[str, Any], result: str = "") -> str | None:
        """Record a tool call. Returns a warning/error message if loop detected."""
        call_hash = _hash_call(name, args)
        result_hash = hashlib.sha256(result.encode()).hexdigest()[:16] if result else ""

        self.history.append(call_hash)
        self.result_hashes.append(result_hash)
        self.total_calls += 1

        # Trim to window
        if len(self.history) > self.window_size:
            self.history = self.history[-self.window_size:]
            self.result_hashes = self.result_hashes[-self.window_size:]

        # 4. Global circuit breaker
        if self.total_calls >= self.breaker_threshold:
            return f"CRITICAL: Global circuit breaker triggered ({self.total_calls} tool calls). Stopping."

        # 1. Generic repeat
        repeat_count = sum(1 for h in self.history if h == call_hash)
        if repeat_count >= self.critical_threshold:
            return f"CRITICAL: Tool '{name}' called {repeat_count} times with same params."
        if repeat_count >= self.warning_threshold:
            return f"WARNING: Tool '{name}' called {repeat_count} times with same params. Consider a different approach."

        # 2. Poll no progress (same tool, same result)
        if len(self.history) >= 4:
            recent = list(zip(self.history[-4:], self.result_hashes[-4:]))
            if all(h == recent[0][0] and r == recent[0][1] for h, r in recent):
                return f"WARNING: Tool '{name}' polling with no progress (4 identical results)."

        # 3. Ping-pong (alternating pattern)
        if len(self.history) >= 6:
            last6 = self.history[-6:]
            if last6[0] == last6[2] == last6[4] and last6[1] == last6[3] == last6[5]:
                if last6[0] != last6[1]:
                    return "WARNING: Ping-pong pattern detected (alternating tool calls with no progress)."

        return None

    def reset(self) -> None:
        self.history.clear()
        self.result_hashes.clear()
        self.total_calls = 0
