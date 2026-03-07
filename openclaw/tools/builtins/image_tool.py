"""Built-in tool: image analysis."""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Any

from openclaw.agent.types import ToolDefinition, ToolParameter, ToolResult

DEFINITION = ToolDefinition(
    name="image",
    description="Analyze an image file or URL. Returns a text description of the image content.",
    parameters=[
        ToolParameter(name="path", description="File path or URL of the image"),
        ToolParameter(name="prompt", description="What to analyze about the image", required=False),
    ],
)


async def execute(args: dict[str, Any], workspace: str = "", **_: Any) -> ToolResult:
    image_path = args.get("path", "")
    prompt = args.get("prompt", "Describe this image in detail.")

    if not image_path:
        return ToolResult(tool_use_id="", content="Error: path is required", is_error=True)

    # For now, return a placeholder — actual implementation would use
    # a vision model via the provider
    return ToolResult(
        tool_use_id="",
        content=f"Image analysis for '{image_path}': [Image analysis requires vision model integration]",
    )
