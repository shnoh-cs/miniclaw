"""Hook system: execute shell commands in response to agent events.

Hooks are defined in config.toml:
    [hooks]
    pre_tool_call = "echo 'Tool: {tool_name}' >> /tmp/agent.log"
    post_tool_call = "echo 'Result: {status}' >> /tmp/agent.log"
    pre_message = ""
    post_message = ""
    on_error = ""

Supported events:
- pre_tool_call: before a tool executes (vars: tool_name, tool_args)
- post_tool_call: after a tool executes (vars: tool_name, status, duration)
- pre_message: before sending user message to model
- post_message: after receiving model response (vars: text_length, tool_count)
- on_error: when an error occurs (vars: error_type, error_message)
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from openclaw.config import HooksConfig

logger = logging.getLogger(__name__)


class HookRunner:
    """Executes shell-command hooks for agent lifecycle events.

    Hooks are fire-and-forget: failures are logged but never block the
    agent loop or raise exceptions to callers.
    """

    def __init__(self, config: HooksConfig) -> None:
        self._config = config
        self._commands: dict[str, str] = {
            "pre_tool_call": config.pre_tool_call,
            "post_tool_call": config.post_tool_call,
            "pre_message": config.pre_message,
            "post_message": config.post_message,
            "on_error": config.on_error,
        }
        self._timeout = config.timeout

    async def fire(self, event_name: str, **kwargs: Any) -> None:
        """Fire a hook for *event_name*, substituting *kwargs* into the
        command template via ``str.format()``.

        If the hook command is empty or the event name is unknown the call
        is silently ignored.  Subprocess errors and timeouts are logged but
        never propagated.
        """
        command_template = self._commands.get(event_name, "")
        if not command_template:
            return

        try:
            command = command_template.format(**kwargs)
        except (KeyError, IndexError, ValueError) as exc:
            logger.warning("Hook %s: failed to format command: %s", event_name, exc)
            return

        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await asyncio.wait_for(proc.communicate(), timeout=self._timeout)

            if proc.returncode != 0:
                logger.warning(
                    "Hook %s exited with code %d", event_name, proc.returncode
                )
        except asyncio.TimeoutError:
            logger.warning("Hook %s timed out after %ds", event_name, self._timeout)
            try:
                proc.kill()  # type: ignore[possibly-undefined]
            except ProcessLookupError:
                pass
        except Exception as exc:
            logger.warning("Hook %s failed: %s", event_name, exc)
