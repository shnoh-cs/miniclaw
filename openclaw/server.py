"""Minimal web server for the OpenClaw agent — send+poll chat + cron.

All endpoints are served under "/" to work behind reverse proxies that
only forward the root path. Query parameter "action" selects the operation:
  GET /                          → chat UI (HTML)
  GET /?action=send&message=...  → send message (fire-and-forget, returns immediately)
  GET /?action=status            → check if agent is processing (JSON)
  GET /?action=history&session_id=... → load previous messages (JSON)
  GET /?action=cron              → cron job list (JSON)
  GET /?action=sessions          → active sessions (JSON)
"""

from __future__ import annotations

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse

from openclaw.agent.api import Agent
from openclaw.agent.loop import run
from openclaw.agent.types import AgentMessage, TextBlock, ToolUseBlock, ToolResultBlock

log = logging.getLogger("openclaw.server")

# ---------------------------------------------------------------------------
# Global agent instance + running task tracker
# ---------------------------------------------------------------------------
_agent: Agent | None = None
_running_tasks: dict[str, asyncio.Task] = {}  # session_id → task


def _get_agent() -> Agent:
    assert _agent is not None, "Agent not initialized"
    return _agent


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    global _agent
    _agent = Agent.from_config()
    log.info("Agent initialized: model=%s, workspace=%s", _agent.config.models.default, _agent.workspace)

    # Start heartbeat + restore persisted cron jobs
    await _agent.start_heartbeat()
    await _agent.restore_cron_jobs()

    yield

    # Shutdown: save cron jobs and stop scheduler
    _agent._save_cron_jobs()
    await _agent.stop_heartbeat()
    log.info("Server shutdown complete")


app = FastAPI(title="OpenClaw", lifespan=lifespan)

# CORS — allow all origins for internal network access
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# ---------------------------------------------------------------------------
# Single root endpoint — dispatches by "action" query param
# ---------------------------------------------------------------------------

@app.get("/")
async def root(request: Request):
    action = request.query_params.get("action", "")

    if action == "send":
        return await _handle_send(request)
    elif action == "status":
        return _handle_status(request)
    elif action == "history":
        return _handle_history(request)
    elif action == "cron":
        return _handle_cron()
    elif action == "sessions":
        return _handle_sessions()
    else:
        # Default: serve HTML UI (no-cache to ensure latest version)
        static_dir = Path(__file__).parent / "static"
        return FileResponse(
            static_dir / "index.html",
            media_type="text/html",
            headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
        )


# ---------------------------------------------------------------------------
# Send (fire-and-forget) + Status
# ---------------------------------------------------------------------------

async def _handle_send(request: Request) -> JSONResponse:
    """Accept a message, start processing in background, return immediately."""
    message = request.query_params.get("message", "").strip()
    session_id = request.query_params.get("session_id", "web")

    if not message:
        return JSONResponse({"error": "message is required"}, status_code=400)

    # Don't accept new messages while already processing for this session
    existing = _running_tasks.get(session_id)
    if existing and not existing.done():
        return JSONResponse({"error": "already processing"}, status_code=409)

    agent = _get_agent()

    async def _run_in_background() -> None:
        try:
            ctx = agent._build_context(session_id)
            await run(ctx, message)
        except Exception:
            log.exception("Background agent run failed for session=%s", session_id)
        finally:
            _running_tasks.pop(session_id, None)

    task = asyncio.create_task(_run_in_background())
    _running_tasks[session_id] = task

    return JSONResponse({"ok": True})


def _handle_status(request: Request) -> JSONResponse:
    """Check if the agent is currently processing for a session."""
    session_id = request.query_params.get("session_id", "web")
    existing = _running_tasks.get(session_id)
    processing = existing is not None and not existing.done()
    return JSONResponse({"processing": processing})


# ---------------------------------------------------------------------------
# History — load previous messages for a session
# ---------------------------------------------------------------------------

def _message_to_dict(msg: AgentMessage) -> dict[str, Any] | None:
    """Convert an AgentMessage to a simple dict for the frontend."""
    text_parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []

    for block in msg.content:
        if isinstance(block, TextBlock) and block.text.strip():
            text_parts.append(block.text)
        elif isinstance(block, ToolUseBlock):
            tool_calls.append({"name": block.name, "args": {k: str(v)[:100] for k, v in (block.input if isinstance(block.input, dict) else {}).items()}})
        elif isinstance(block, ToolResultBlock):
            ok = not block.is_error
            preview = (block.content[:200].replace("\n", " ") if isinstance(block.content, str) else "")
            tool_calls.append({"name": "result", "ok": ok, "preview": preview})

    text = "\n".join(text_parts)
    if not text and not tool_calls:
        return None

    return {
        "role": msg.role,
        "text": text,
        "tool_calls": tool_calls if tool_calls else None,
    }


def _handle_history(request: Request) -> JSONResponse:
    session_id = request.query_params.get("session_id", "web")
    agent = _get_agent()
    session = agent._get_session(session_id)
    session.load()

    messages: list[dict[str, Any]] = []
    for msg in session.messages:
        d = _message_to_dict(msg)
        if d:
            messages.append(d)

    # Also include compaction summary if present
    compaction_summary = session.latest_compaction_summary

    return JSONResponse({
        "session_id": session_id,
        "messages": messages,
        "compaction_summary": compaction_summary,
    })


# ---------------------------------------------------------------------------
# Info endpoints
# ---------------------------------------------------------------------------

def _handle_cron() -> JSONResponse:
    agent = _get_agent()
    return JSONResponse(agent.scheduler.status())


def _handle_sessions() -> JSONResponse:
    agent = _get_agent()
    result = [{"id": sid, "messages": len(mgr.messages)} for sid, mgr in agent._sessions.items()]
    return JSONResponse(result)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser(description="OpenClaw web server")
    parser.add_argument("-p", "--port", type=int, default=8089, help="Port (default: 8089)")
    parser.add_argument("--host", default="0.0.0.0", help="Host (default: 0.0.0.0)")
    parser.add_argument("--reload", action="store_true", help="Auto-reload on file changes")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    uvicorn.run("openclaw.server:app", host=args.host, port=args.port, log_level="info", reload=args.reload)


if __name__ == "__main__":
    main()
