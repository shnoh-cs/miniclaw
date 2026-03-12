"""Cron job persistence — save/load/restore from JSON file."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from openclaw.cron import CronScheduler, Schedule, ScheduleKind

log = logging.getLogger(__name__)

# System tasks created by start_heartbeat() — skip when saving
_SYSTEM_TASKS = frozenset({"model_ping", "memory_check", "heartbeat_file"})


@dataclass
class JobRecord:
    """Serializable cron job metadata."""

    name: str
    task: str
    schedule: dict[str, Any]
    one_shot: bool = False
    created_at: float = 0.0
    last_run: float = 0.0
    run_count: int = 0
    reply_to: str = ""

    def to_schedule(self) -> Schedule:
        return Schedule(
            kind=ScheduleKind(self.schedule["kind"]),
            interval_seconds=self.schedule.get("interval_seconds", 0.0),
            cron_expr=self.schedule.get("cron_expr", ""),
            timezone=self.schedule.get("timezone", ""),
            at=self.schedule.get("at", ""),
        )


def save_jobs(
    scheduler: CronScheduler,
    task_descriptions: dict[str, str],
    path: Path,
    *,
    reply_to_map: dict[str, str] | None = None,
) -> None:
    """Save user-created cron jobs to JSON file."""
    reply_to_map = reply_to_map or {}
    records: list[dict[str, Any]] = []
    for info in scheduler.status():
        name = info["name"]
        if name in _SYSTEM_TASKS or name not in task_descriptions:
            continue
        task_obj = scheduler._tasks.get(name)
        if task_obj is None:
            continue
        records.append({
            "name": name,
            "task": task_descriptions[name],
            "schedule": {
                "kind": task_obj.schedule.kind.value,
                "interval_seconds": task_obj.schedule.interval_seconds,
                "cron_expr": task_obj.schedule.cron_expr,
                "timezone": task_obj.schedule.timezone,
                "at": task_obj.schedule.at,
            },
            "one_shot": task_obj.one_shot,
            "created_at": getattr(task_obj, "_created_at", time.time()),
            "last_run": task_obj.last_run,
            "run_count": task_obj.run_count,
            "reply_to": reply_to_map.get(name, ""),
        })

    path.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("Saved %d cron jobs to %s", len(records), path)


def load_jobs(path: Path) -> list[JobRecord]:
    """Load cron job records from JSON file."""
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        records = [
            JobRecord(
                name=r["name"],
                task=r["task"],
                schedule=r["schedule"],
                one_shot=r.get("one_shot", False),
                created_at=r.get("created_at", 0.0),
                last_run=r.get("last_run", 0.0),
                run_count=r.get("run_count", 0),
                reply_to=r.get("reply_to", ""),
            )
            for r in data
        ]
        log.info("Loaded %d cron jobs from %s", len(records), path)
        return records
    except Exception as e:
        log.warning("Failed to load cron jobs from %s: %s", path, e)
        return []


def check_missed_jobs(records: list[JobRecord]) -> list[str]:
    """Return names of cron-expression jobs that missed a scheduled run."""
    missed: list[str] = []
    now = time.time()

    for rec in records:
        if rec.schedule.get("kind") != "cron" or not rec.schedule.get("cron_expr"):
            continue
        if rec.one_shot and rec.run_count > 0:
            continue
        if rec.last_run <= 0:
            # Never ran — new job, wait for next scheduled time
            continue

        # Check if any scheduled time was missed since last_run
        try:
            from croniter import croniter
            from datetime import datetime, timezone as dt_tz

            last_dt = datetime.fromtimestamp(rec.last_run, tz=dt_tz.utc)
            cron = croniter(rec.schedule["cron_expr"], last_dt)
            next_dt = cron.get_next(datetime)
            if next_dt.timestamp() <= now:
                missed.append(rec.name)
        except Exception:
            pass

    # Also check one-shot "at" jobs that never ran
    for rec in records:
        if rec.schedule.get("kind") != "at" or rec.run_count > 0:
            continue
        at_str = rec.schedule.get("at", "")
        if not at_str:
            continue
        try:
            from dateutil import parser as dateutil_parser
            target = dateutil_parser.parse(at_str)
            if target.timestamp() <= now:
                missed.append(rec.name)
        except Exception:
            pass

    return missed
