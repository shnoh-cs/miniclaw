"""Built-in tool: write/create files."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from openclaw.agent.types import ToolDefinition, ToolParameter, ToolResult

DEFINITION = ToolDefinition(
    name="write",
    description="Create or overwrite a file with the given content.",
    parameters=[
        ToolParameter(name="file_path", description="Absolute or workspace-relative path"),
        ToolParameter(name="content", description="Content to write to the file"),
    ],
)


async def execute(args: dict[str, Any], workspace: str = "") -> ToolResult:
    file_path = args.get("file_path", "")
    content = args.get("content", "")

    if not file_path:
        return ToolResult(tool_use_id="", content="Error: file_path is required", is_error=True)

    path = Path(file_path)
    if not path.is_absolute() and workspace:
        path = Path(workspace) / path

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return ToolResult(tool_use_id="", content=f"File written: {path} ({len(content)} chars)")
    except OSError as e:
        return ToolResult(tool_use_id="", content=f"Error writing file: {e}", is_error=True)
