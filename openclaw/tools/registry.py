"""Tool registry: registration, lookup, schema conversion, and result guarding.

Loop detection and truncation logic are split into ``loop_detector`` and
``truncation`` submodules.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from openclaw.agent.types import ToolDefinition, ToolResult

# Re-export public API for backward compatibility
from openclaw.tools.loop_detector import (  # noqa: F401
    LoopDetectionResult,
    ToolLoopDetector,
)
from openclaw.tools.truncation import (  # noqa: F401
    HARD_LIMIT_CHARS,
    MIDDLE_OMISSION_MARKER,
    MISSING_TOOL_RESULT_PLACEHOLDER,
    SESSION_CAP_NOTICE,
    SESSION_HARD_CAP,
    TRUNCATION_SUFFIX,
    cap_tool_result_for_session,
    synthesize_missing_tool_result,
    truncate_tool_result,
)

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
