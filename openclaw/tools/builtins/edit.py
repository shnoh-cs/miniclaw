"""Built-in tool: edit files (find and replace)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from openclaw.agent.types import ToolDefinition, ToolParameter, ToolResult

DEFINITION = ToolDefinition(
    name="edit",
    description="Edit a file by replacing an exact string match with new content.",
    parameters=[
        ToolParameter(name="file_path", description="Path to the file to edit"),
        ToolParameter(name="old_string", description="Exact text to find and replace"),
        ToolParameter(name="new_string", description="Replacement text"),
        ToolParameter(name="replace_all", type="boolean", description="Replace all occurrences", required=False),
    ],
)


async def execute(args: dict[str, Any], workspace: str = "") -> ToolResult:
    file_path = args.get("file_path", "")
    old_string = args.get("old_string", "")
    new_string = args.get("new_string", "")
    replace_all = bool(args.get("replace_all", False))

    if not file_path or not old_string:
        return ToolResult(tool_use_id="", content="Error: file_path and old_string are required", is_error=True)

    path = Path(file_path)
    if not path.is_absolute() and workspace:
        path = Path(workspace) / path

    if not path.exists():
        return ToolResult(tool_use_id="", content=f"Error: File not found: {path}", is_error=True)

    try:
        content = path.read_text(encoding="utf-8")
    except OSError as e:
        return ToolResult(tool_use_id="", content=f"Error reading file: {e}", is_error=True)

    count = content.count(old_string)
    if count == 0:
        return ToolResult(
            tool_use_id="",
            content=f"Error: old_string not found in {path}",
            is_error=True,
        )

    if count > 1 and not replace_all:
        return ToolResult(
            tool_use_id="",
            content=f"Error: old_string found {count} times. Use replace_all=true or provide more context.",
            is_error=True,
        )

    if replace_all:
        new_content = content.replace(old_string, new_string)
        replaced = count
    else:
        new_content = content.replace(old_string, new_string, 1)
        replaced = 1

    try:
        path.write_text(new_content, encoding="utf-8")
        return ToolResult(tool_use_id="", content=f"Edited {path}: {replaced} replacement(s)")
    except OSError as e:
        return ToolResult(tool_use_id="", content=f"Error writing file: {e}", is_error=True)
