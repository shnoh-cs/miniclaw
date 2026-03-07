"""Built-in tool: read PDF documents."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from openclaw.agent.types import ToolDefinition, ToolParameter, ToolResult

DEFINITION = ToolDefinition(
    name="pdf",
    description="Read and extract text from a PDF file.",
    parameters=[
        ToolParameter(name="file_path", description="Path to the PDF file"),
        ToolParameter(name="pages", description="Page range (e.g. '1-5', '3', '10-20')", required=False),
    ],
)


async def execute(args: dict[str, Any], workspace: str = "") -> ToolResult:
    file_path = args.get("file_path", "")
    pages_str = args.get("pages", "")

    if not file_path:
        return ToolResult(tool_use_id="", content="Error: file_path is required", is_error=True)

    path = Path(file_path)
    if not path.is_absolute() and workspace:
        path = Path(workspace) / path

    if not path.exists():
        return ToolResult(tool_use_id="", content=f"Error: File not found: {path}", is_error=True)

    try:
        from PyPDF2 import PdfReader
    except ImportError:
        return ToolResult(
            tool_use_id="",
            content="Error: PyPDF2 is required. Install with: pip install PyPDF2",
            is_error=True,
        )

    try:
        reader = PdfReader(str(path))
        total_pages = len(reader.pages)
    except Exception as e:
        return ToolResult(tool_use_id="", content=f"Error reading PDF: {e}", is_error=True)

    # Parse page range
    start, end = 0, total_pages
    if pages_str:
        try:
            if "-" in pages_str:
                parts = pages_str.split("-")
                start = int(parts[0]) - 1
                end = int(parts[1])
            else:
                start = int(pages_str) - 1
                end = start + 1
        except (ValueError, IndexError):
            return ToolResult(tool_use_id="", content=f"Error: Invalid page range: {pages_str}", is_error=True)

    start = max(0, start)
    end = min(total_pages, end)

    if end - start > 20:
        return ToolResult(
            tool_use_id="",
            content=f"Error: Max 20 pages per request. Total pages: {total_pages}",
            is_error=True,
        )

    parts: list[str] = []
    for i in range(start, end):
        page = reader.pages[i]
        text = page.extract_text() or ""
        parts.append(f"--- Page {i + 1} ---\n{text}")

    output = "\n\n".join(parts)
    if not output.strip():
        output = "(No text content extracted from PDF)"

    return ToolResult(tool_use_id="", content=output)
