"""Built-in tools: memory_search and memory_save."""

from __future__ import annotations

from typing import Any

from openclaw.agent.types import ToolDefinition, ToolParameter, ToolResult

SEARCH_DEFINITION = ToolDefinition(
    name="memory_search",
    description="Search stored memories using semantic (vector + keyword) hybrid search. Returns relevant snippets with file paths and scores.",
    parameters=[
        ToolParameter(name="query", description="The search query"),
        ToolParameter(name="max_results", type="integer", description="Maximum results to return (default 10)", required=False),
    ],
)

SAVE_DEFINITION = ToolDefinition(
    name="memory_save",
    description="Save important information to memory for future retrieval. Writes to memory files in the workspace.",
    parameters=[
        ToolParameter(name="content", description="The content to save"),
        ToolParameter(name="file", description="Target file path relative to memory/ (e.g. '2026-03-08.md')", required=False),
    ],
)


# These are stubs — actual implementation calls into memory.search module
# They get wired up in the agent loop with the actual memory manager

async def execute_search(args: dict[str, Any], **_: Any) -> ToolResult:
    """Stub — replaced at runtime by agent loop."""
    return ToolResult(
        tool_use_id="",
        content="Memory search not configured",
        is_error=True,
    )


async def execute_save(args: dict[str, Any], **_: Any) -> ToolResult:
    """Stub — replaced at runtime by agent loop."""
    return ToolResult(
        tool_use_id="",
        content="Memory save not configured",
        is_error=True,
    )
