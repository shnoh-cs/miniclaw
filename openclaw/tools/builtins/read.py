"""Built-in tool: read files."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from openclaw.agent.types import ToolDefinition, ToolParameter, ToolResult

DEFINITION = ToolDefinition(
    name="read",
    description="Read the contents of a file. Supports text files and returns line-numbered output.",
    parameters=[
        ToolParameter(name="file_path", description="Absolute or workspace-relative path to read"),
        ToolParameter(name="offset", type="integer", description="Line number to start from (1-based)", required=False),
        ToolParameter(name="limit", type="integer", description="Max lines to read", required=False),
    ],
)


async def execute(args: dict[str, Any], workspace: str = "") -> ToolResult:
    file_path = args.get("file_path", "")
    offset = int(args.get("offset", 1))
    limit = int(args.get("limit", 2000))

    if not file_path:
        return ToolResult(tool_use_id="", content="Error: file_path is required", is_error=True)

    path = Path(file_path)
    if not path.is_absolute() and workspace:
        path = Path(workspace) / path

    if not path.exists():
        return ToolResult(tool_use_id="", content=f"Error: File not found: {path}", is_error=True)

    if not path.is_file():
        return ToolResult(tool_use_id="", content=f"Error: Not a file: {path}", is_error=True)

    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        return ToolResult(tool_use_id="", content=f"Error reading file: {e}", is_error=True)

    lines = text.splitlines()
    start = max(0, offset - 1)
    end = start + limit
    selected = lines[start:end]

    # Format with line numbers
    numbered = []
    for i, line in enumerate(selected, start=start + 1):
        truncated = line[:2000] if len(line) > 2000 else line
        numbered.append(f"{i:>6}\t{truncated}")

    output = "\n".join(numbered)
    if end < len(lines):
        output += f"\n\n... ({len(lines) - end} more lines)"

    return ToolResult(tool_use_id="", content=output)
