"""Built-in tool: execute shell commands."""

from __future__ import annotations

import asyncio
from typing import Any

from openclaw.agent.types import ToolDefinition, ToolParameter, ToolResult

DEFINITION = ToolDefinition(
    name="bash",
    description="Execute a shell command and return its output. Use for system operations, git, package managers, etc.",
    parameters=[
        ToolParameter(name="command", description="The shell command to execute"),
        ToolParameter(name="timeout", type="integer", description="Timeout in seconds (default 120)", required=False),
        ToolParameter(name="cwd", description="Working directory for the command", required=False),
    ],
)


async def execute(args: dict[str, Any], workspace: str = "") -> ToolResult:
    command = args.get("command", "")
    timeout = int(args.get("timeout", 120))
    cwd = args.get("cwd", workspace) or workspace or None

    if not command:
        return ToolResult(tool_use_id="", content="Error: command is required", is_error=True)

    try:
        proc = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=cwd,
        )
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(), timeout=timeout
        )
    except asyncio.TimeoutError:
        proc.kill()
        return ToolResult(
            tool_use_id="",
            content=f"Error: Command timed out after {timeout}s",
            is_error=True,
        )
    except OSError as e:
        return ToolResult(tool_use_id="", content=f"Error: {e}", is_error=True)

    output_parts: list[str] = []
    if stdout:
        output_parts.append(stdout.decode("utf-8", errors="replace"))
    if stderr:
        output_parts.append(f"STDERR:\n{stderr.decode('utf-8', errors='replace')}")

    output = "\n".join(output_parts) if output_parts else "(no output)"

    if proc.returncode != 0:
        output = f"Exit code: {proc.returncode}\n{output}"
        return ToolResult(tool_use_id="", content=output, is_error=True)

    return ToolResult(tool_use_id="", content=output)
