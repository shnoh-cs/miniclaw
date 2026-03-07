"""Built-in tool: manage background processes."""

from __future__ import annotations

import asyncio
import os
import signal
from dataclasses import dataclass, field
from typing import Any

from openclaw.agent.types import ToolDefinition, ToolParameter, ToolResult

DEFINITION = ToolDefinition(
    name="process",
    description="Manage background shell processes. Actions: list, poll, log, write, kill, clear.",
    parameters=[
        ToolParameter(
            name="action",
            description="Action to perform",
            enum=["start", "list", "poll", "log", "write", "kill"],
        ),
        ToolParameter(name="command", description="Command to start (for 'start' action)", required=False),
        ToolParameter(name="pid", type="integer", description="Process ID (for poll/log/write/kill)", required=False),
        ToolParameter(name="input", description="Input to write to process stdin (for 'write')", required=False),
        ToolParameter(name="cwd", description="Working directory", required=False),
    ],
)


@dataclass
class BackgroundProcess:
    pid: int
    command: str
    process: asyncio.subprocess.Process
    stdout_buffer: list[str] = field(default_factory=list)
    stderr_buffer: list[str] = field(default_factory=list)


# Global process registry
_processes: dict[int, BackgroundProcess] = {}


async def execute(args: dict[str, Any], workspace: str = "") -> ToolResult:
    action = args.get("action", "")
    cwd = args.get("cwd", workspace) or workspace or None

    if action == "start":
        command = args.get("command", "")
        if not command:
            return ToolResult(tool_use_id="", content="Error: command required for start", is_error=True)

        proc = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            stdin=asyncio.subprocess.PIPE,
            cwd=cwd,
        )
        bp = BackgroundProcess(pid=proc.pid, command=command, process=proc)
        _processes[proc.pid] = bp
        return ToolResult(tool_use_id="", content=f"Started process PID={proc.pid}: {command}")

    if action == "list":
        if not _processes:
            return ToolResult(tool_use_id="", content="No background processes")
        lines = [f"PID={p.pid} | {p.command}" for p in _processes.values()]
        return ToolResult(tool_use_id="", content="\n".join(lines))

    pid = int(args.get("pid", 0))
    bp = _processes.get(pid)
    if not bp:
        return ToolResult(tool_use_id="", content=f"Error: No process with PID={pid}", is_error=True)

    if action == "poll":
        returncode = bp.process.returncode
        if returncode is None:
            return ToolResult(tool_use_id="", content=f"PID={pid} still running")
        return ToolResult(tool_use_id="", content=f"PID={pid} exited with code {returncode}")

    if action == "log":
        # Read available stdout
        try:
            stdout = await asyncio.wait_for(bp.process.stdout.read(4096), timeout=1.0)
            text = stdout.decode("utf-8", errors="replace") if stdout else "(no output)"
        except asyncio.TimeoutError:
            text = "(no new output)"
        return ToolResult(tool_use_id="", content=text)

    if action == "write":
        input_data = args.get("input", "")
        if bp.process.stdin:
            bp.process.stdin.write(input_data.encode())
            await bp.process.stdin.drain()
            return ToolResult(tool_use_id="", content=f"Wrote to PID={pid}")
        return ToolResult(tool_use_id="", content="Error: stdin not available", is_error=True)

    if action == "kill":
        try:
            bp.process.kill()
            del _processes[pid]
            return ToolResult(tool_use_id="", content=f"Killed PID={pid}")
        except ProcessLookupError:
            del _processes[pid]
            return ToolResult(tool_use_id="", content=f"PID={pid} already exited")

    return ToolResult(tool_use_id="", content=f"Error: Unknown action '{action}'", is_error=True)
