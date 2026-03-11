"""Cron / heartbeat scheduler for periodic agent tasks.

Supports:
- Heartbeat: periodic health checks (model connectivity, memory index, etc.)
- Scheduled tasks: cron-like jobs that run at intervals
- One-shot timers: delayed execution of a single task

All tasks run as asyncio background tasks and are non-blocking.
Failures are logged but never crash the agent.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Coroutine

logger = logging.getLogger(__name__)


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class ScheduledTask:
    """A single scheduled task."""

    name: str
    callback: Callable[..., Coroutine[Any, Any, None]]
    interval_seconds: float
    one_shot: bool = False
    status: TaskStatus = TaskStatus.PENDING
    last_run: float = 0.0
    run_count: int = 0
    last_error: str | None = None
    _task: asyncio.Task[None] | None = field(default=None, repr=False)


class CronScheduler:
    """Lightweight async scheduler for periodic agent tasks.

    Usage:
        scheduler = CronScheduler()

        # Register a heartbeat (runs every 60s)
        scheduler.register("heartbeat", check_health, interval=60)

        # Register a one-shot timer
        scheduler.register("cleanup", do_cleanup, interval=300, one_shot=True)

        # Start all tasks
        await scheduler.start()

        # Later...
        await scheduler.stop()
    """

    def __init__(self) -> None:
        self._tasks: dict[str, ScheduledTask] = {}
        self._running = False

    def register(
        self,
        name: str,
        callback: Callable[..., Coroutine[Any, Any, None]],
        interval: float = 60.0,
        one_shot: bool = False,
    ) -> None:
        """Register a new scheduled task."""
        self._tasks[name] = ScheduledTask(
            name=name,
            callback=callback,
            interval_seconds=interval,
            one_shot=one_shot,
        )

    def unregister(self, name: str) -> bool:
        """Unregister and cancel a task."""
        task = self._tasks.pop(name, None)
        if task and task._task:
            task._task.cancel()
            return True
        return False

    async def start(self) -> None:
        """Start all registered tasks."""
        if self._running:
            return
        self._running = True

        for name, task in self._tasks.items():
            if task.status in (TaskStatus.PENDING, TaskStatus.COMPLETED):
                task._task = asyncio.create_task(
                    self._run_loop(task), name=f"cron-{name}"
                )
                task.status = TaskStatus.RUNNING
                logger.info(
                    "Cron: started '%s' (interval=%ds, one_shot=%s)",
                    name, task.interval_seconds, task.one_shot,
                )

    async def stop(self) -> None:
        """Stop all running tasks gracefully."""
        self._running = False
        for task in self._tasks.values():
            if task._task and not task._task.done():
                task._task.cancel()
                task.status = TaskStatus.CANCELLED

        # Wait for cancellation
        pending = [t._task for t in self._tasks.values() if t._task and not t._task.done()]
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)

        logger.info("Cron: all tasks stopped")

    def status(self) -> list[dict[str, Any]]:
        """Get status of all scheduled tasks."""
        return [
            {
                "name": t.name,
                "status": t.status.value,
                "interval": t.interval_seconds,
                "one_shot": t.one_shot,
                "run_count": t.run_count,
                "last_run": t.last_run,
                "last_error": t.last_error,
            }
            for t in self._tasks.values()
        ]

    async def _run_loop(self, task: ScheduledTask) -> None:
        """Internal loop for a single task."""
        try:
            while self._running:
                await asyncio.sleep(task.interval_seconds)

                if not self._running:
                    break

                try:
                    task.last_run = time.time()
                    await task.callback()
                    task.run_count += 1
                    task.last_error = None

                    if task.one_shot:
                        task.status = TaskStatus.COMPLETED
                        logger.info("Cron: one-shot '%s' completed", task.name)
                        break

                except Exception as exc:
                    task.last_error = str(exc)
                    task.run_count += 1
                    logger.warning("Cron: '%s' failed: %s", task.name, exc)

                    if task.one_shot:
                        task.status = TaskStatus.FAILED
                        break

        except asyncio.CancelledError:
            task.status = TaskStatus.CANCELLED


# ---------------------------------------------------------------------------
# Built-in heartbeat tasks
# ---------------------------------------------------------------------------


async def heartbeat_model_ping(provider: Any, model: str) -> None:
    """Ping the model endpoint to verify connectivity."""
    try:
        response = await provider.client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=1,
        )
        logger.debug("Heartbeat: model '%s' responded OK", model)
    except Exception as exc:
        logger.warning("Heartbeat: model '%s' unreachable: %s", model, exc)
        raise


async def heartbeat_from_file(
    heartbeat_path: str,
    provider: Any,
    model: str,
    workspace_dir: str,
    *,
    agent: Any | None = None,
) -> None:
    """Execute heartbeat instructions from a HEARTBEAT.md file.

    If an Agent instance is provided, the heartbeat runs through the full
    agent loop so that tool calls (bash, web_fetch, etc.) can actually execute.
    Otherwise, falls back to a plain completion (no tool execution).
    """
    from pathlib import Path

    hb_file = Path(heartbeat_path)
    if not hb_file.exists():
        return

    content = hb_file.read_text(encoding="utf-8", errors="replace").strip()
    if not content:
        return

    prompt = (
        "You are running a scheduled heartbeat check. "
        "Review the instructions below and perform any due actions. "
        "If no actions are due right now, respond with NO_REPLY.\n\n"
        f"Current heartbeat instructions:\n\n{content}"
    )

    # Prefer full agent loop (tools available)
    if agent is not None:
        try:
            result = await agent.run(
                prompt,
                session_id="heartbeat",
            )
            text = result.text or ""
            if text.strip().upper() != "NO_REPLY":
                logger.info("Heartbeat action: %s", text[:200])
            return
        except Exception as exc:
            logger.warning("Heartbeat agent loop failed, falling back: %s", exc)

    # Fallback: plain completion (no tool execution)
    try:
        response = await provider.client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": (
                    "You are a personal assistant running a scheduled heartbeat check. "
                    "Review the heartbeat instructions below and perform any due actions. "
                    "If no actions are due right now, respond with NO_REPLY."
                )},
                {"role": "user", "content": f"Current heartbeat instructions:\n\n{content}"},
            ],
            max_tokens=2048,
        )
        result_text = response.choices[0].message.content or ""
        if result_text.strip().upper() != "NO_REPLY":
            logger.info("Heartbeat action (no tools): %s", result_text[:200])
    except Exception as exc:
        logger.warning("Heartbeat file execution failed: %s", exc)


async def heartbeat_memory_check(memory_dir: str) -> None:
    """Check memory store health (DB accessible, index intact)."""
    from pathlib import Path

    db_path = Path(memory_dir) / "memory.sqlite"
    if not db_path.exists():
        logger.debug("Heartbeat: no memory DB at %s (OK for fresh install)", db_path)
        return

    import sqlite3
    try:
        conn = sqlite3.connect(str(db_path))
        cursor = conn.execute("SELECT count(*) FROM chunks")
        count = cursor.fetchone()[0]
        conn.close()
        logger.debug("Heartbeat: memory store has %d chunks", count)
    except Exception as exc:
        logger.warning("Heartbeat: memory DB error: %s", exc)
        raise
