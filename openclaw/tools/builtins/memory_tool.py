"""Built-in tools: memory_search, memory_save, and memory_get."""

from __future__ import annotations

from pathlib import Path
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

MEMORY_GET_DEFINITION = ToolDefinition(
    name="memory_get",
    description="Retrieve specific lines from a memory file. Use after memory_search to read targeted content around a result.",
    parameters=[
        ToolParameter(name="path", description="Absolute or workspace-relative file path"),
        ToolParameter(name="line_start", type="integer", description="First line number to read (1-based)"),
        ToolParameter(name="line_end", type="integer", description="Last line number to read (1-based, inclusive)"),
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


async def execute_memory_get(args: dict[str, Any], **_: Any) -> ToolResult:
    """Read a specific line range from a memory file."""
    file_path = args.get("path", "")
    line_start = int(args.get("line_start", 1))
    line_end = int(args.get("line_end", line_start + 50))

    if not file_path:
        return ToolResult(tool_use_id="", content="Missing 'path' parameter", is_error=True)

    p = Path(file_path)
    if not p.exists():
        return ToolResult(tool_use_id="", content=f"File not found: {file_path}", is_error=True)

    # Clamp to valid range
    if line_start < 1:
        line_start = 1
    if line_end < line_start:
        line_end = line_start

    try:
        all_lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as e:
        return ToolResult(tool_use_id="", content=f"Error reading file: {e}", is_error=True)

    # Convert to 0-based indices, clamp to file length
    start_idx = line_start - 1
    end_idx = min(line_end, len(all_lines))

    if start_idx >= len(all_lines):
        return ToolResult(
            tool_use_id="",
            content=f"line_start ({line_start}) exceeds file length ({len(all_lines)} lines)",
            is_error=True,
        )

    selected = all_lines[start_idx:end_idx]
    # Format with line numbers
    numbered = [f"{start_idx + i + 1:>6}\t{line}" for i, line in enumerate(selected)]
    header = f"# {file_path} (lines {line_start}-{end_idx})\n"
    return ToolResult(tool_use_id="", content=header + "\n".join(numbered))
