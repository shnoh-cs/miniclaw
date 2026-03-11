"""Minimal web server for the OpenClaw agent — SSE streaming chat + cron."""

from __future__ import annotations

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse

from openclaw.agent.api import Agent
from openclaw.agent.loop import run
from openclaw.agent.types import ToolResult

log = logging.getLogger("openclaw.server")

# ---------------------------------------------------------------------------
# Global agent instance (created on startup)
# ---------------------------------------------------------------------------
_agent: Agent | None = None


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


# ---------------------------------------------------------------------------
# SSE chat endpoint
# ---------------------------------------------------------------------------

@app.post("/api/chat")
async def chat(request: Request) -> StreamingResponse:
    body = await request.json()
    message = body.get("message", "").strip()
    session_id = body.get("session_id", "web")

    if not message:
        return JSONResponse({"error": "message is required"}, status_code=400)

    agent = _get_agent()

    async def event_stream() -> AsyncIterator[str]:
        queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()

        def on_stream(text: str) -> None:
            queue.put_nowait({"event": "text", "data": {"text": text}})

        def on_tool_start(name: str, args: dict) -> None:
            queue.put_nowait({
                "event": "tool_start",
                "data": {"name": name, "args": {k: str(v)[:100] for k, v in args.items()}},
            })

        def on_tool_end(name: str, result: ToolResult) -> None:
            queue.put_nowait({
                "event": "tool_end",
                "data": {
                    "name": name,
                    "ok": not result.is_error,
                    "preview": result.content[:200].replace("\n", " "),
                },
            })

        ctx = agent._build_context(session_id)
        ctx.on_stream = on_stream
        ctx.on_tool_start = on_tool_start
        ctx.on_tool_end = on_tool_end

        async def run_agent() -> RunResult:
            return await run(ctx, message)

        task = asyncio.create_task(run_agent())

        # Yield SSE events as they arrive
        while not task.done():
            try:
                item = await asyncio.wait_for(queue.get(), timeout=0.2)
                if item is None:
                    break
                yield f"event: {item['event']}\ndata: {json.dumps(item['data'], ensure_ascii=False)}\n\n"
            except asyncio.TimeoutError:
                # Send keepalive comment to prevent proxy timeout
                yield ": keepalive\n\n"

        # Drain remaining events
        while not queue.empty():
            item = queue.get_nowait()
            if item is not None:
                yield f"event: {item['event']}\ndata: {json.dumps(item['data'], ensure_ascii=False)}\n\n"

        # Final result
        try:
            result = task.result()
            yield f"event: done\ndata: {json.dumps({'text': result.text or '', 'error': result.error, 'compacted': result.compacted}, ensure_ascii=False)}\n\n"
        except Exception as exc:
            yield f"event: error\ndata: {json.dumps({'error': str(exc)}, ensure_ascii=False)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# ---------------------------------------------------------------------------
# Info endpoints
# ---------------------------------------------------------------------------

@app.get("/api/sessions")
async def sessions() -> JSONResponse:
    agent = _get_agent()
    result = []
    for sid, mgr in agent._sessions.items():
        result.append({"id": sid, "messages": len(mgr.messages)})
    return JSONResponse(result)


@app.get("/api/cron")
async def cron_jobs() -> JSONResponse:
    agent = _get_agent()
    return JSONResponse(agent.scheduler.status())


# ---------------------------------------------------------------------------
# Static UI
# ---------------------------------------------------------------------------

@app.get("/")
async def index() -> FileResponse:
    static_dir = Path(__file__).parent / "static"
    return FileResponse(static_dir / "index.html", media_type="text/html")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser(description="OpenClaw web server")
    parser.add_argument("-p", "--port", type=int, default=8089, help="Port (default: 8089)")
    parser.add_argument("--host", default="0.0.0.0", help="Host (default: 0.0.0.0)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    uvicorn.run("openclaw.server:app", host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
