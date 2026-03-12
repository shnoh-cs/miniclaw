"""Rocket.Chat integration — REST API polling bridge."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import httpx

from openclaw.config import RocketChatConfig

log = logging.getLogger("openclaw.rocketchat")


class RocketChatClient:
    """Thin async wrapper around the Rocket.Chat REST API."""

    def __init__(self, base_url: str) -> None:
        self._base_url = base_url.rstrip("/")
        self._client = httpx.AsyncClient(base_url=self._base_url, timeout=30.0)
        self.user_id: str = ""
        self.auth_token: str = ""

    async def login(self, user: str, password: str) -> None:
        resp = await self._client.post(
            "/api/v1/login", json={"user": user, "password": password}
        )
        resp.raise_for_status()
        data = resp.json()["data"]
        self.user_id = data["userId"]
        self.auth_token = data["authToken"]
        self._client.headers.update({
            "X-Auth-Token": self.auth_token,
            "X-User-Id": self.user_id,
        })
        log.info("Logged in as %s (uid=%s)", user, self.user_id)

    async def send_message(self, room_id: str, text: str) -> None:
        # Rocket.Chat has ~48KB message limit; split at 4000 chars for readability
        for i in range(0, len(text), 4000):
            chunk = text[i : i + 4000]
            resp = await self._client.post(
                "/api/v1/chat.sendMessage",
                json={"message": {"rid": room_id, "msg": chunk}},
            )
            if resp.status_code == 401:
                raise PermissionError("Auth token expired")
            resp.raise_for_status()

    async def get_history(self, room_id: str, oldest: str) -> list[dict[str, Any]]:
        resp = await self._client.get(
            "/api/v1/channels.history",
            params={"roomId": room_id, "oldest": oldest, "count": 50},
        )
        if resp.status_code == 401:
            raise PermissionError("Auth token expired")
        resp.raise_for_status()
        return resp.json().get("messages", [])

    async def get_dm_history(self, room_id: str, oldest: str) -> list[dict[str, Any]]:
        resp = await self._client.get(
            "/api/v1/im.history",
            params={"roomId": room_id, "oldest": oldest, "count": 50},
        )
        if resp.status_code == 401:
            raise PermissionError("Auth token expired")
        resp.raise_for_status()
        return resp.json().get("messages", [])

    async def resolve_channel_id(self, name: str) -> str:
        resp = await self._client.get(
            "/api/v1/channels.info", params={"roomName": name}
        )
        resp.raise_for_status()
        return resp.json()["channel"]["_id"]

    async def get_dm_rooms(self) -> list[dict[str, Any]]:
        resp = await self._client.get(
            "/api/v1/im.list", params={"count": 100}
        )
        resp.raise_for_status()
        return resp.json().get("ims", [])

    async def close(self) -> None:
        await self._client.aclose()


class RocketChatBridge:
    """Polls Rocket.Chat for new messages and routes them to the agent."""

    def __init__(self, agent: Any, config: RocketChatConfig) -> None:
        from openclaw.agent.api import Agent

        self._agent: Agent = agent
        self._config = config
        self._client = RocketChatClient(config.url)
        self._poll_task: asyncio.Task | None = None
        self._running = False
        # room_id → ISO timestamp of last seen message
        self._last_ts: dict[str, str] = {}
        # room_id → True if agent is currently processing
        self._processing: dict[str, bool] = {}
        # channel name → room_id
        self._channel_ids: dict[str, str] = {}

    async def start(self) -> None:
        await self._client.login(self._config.user, self._config.password)

        # Resolve channel names to IDs
        for name in self._config.channels:
            try:
                rid = await self._client.resolve_channel_id(name)
                self._channel_ids[name] = rid
                log.info("Monitoring channel: %s (%s)", name, rid)
            except Exception:
                log.warning("Failed to resolve channel: %s", name)

        self._running = True
        self._poll_task = asyncio.create_task(self._poll_loop(), name="rc-poll")
        log.info("Rocket.Chat bridge started")

    async def stop(self) -> None:
        self._running = False
        if self._poll_task:
            self._poll_task.cancel()
            try:
                await self._poll_task
            except asyncio.CancelledError:
                pass
        await self._client.close()
        log.info("Rocket.Chat bridge stopped")

    async def send_notification(self, text: str) -> None:
        """Send a message to the configured notify channel."""
        channel = self._config.notify_channel
        if not channel:
            return
        rid = self._channel_ids.get(channel)
        if not rid:
            try:
                rid = await self._client.resolve_channel_id(channel)
                self._channel_ids[channel] = rid
            except Exception:
                log.warning("Cannot resolve notify channel: %s", channel)
                return
        await self._client.send_message(rid, text)

    async def _poll_loop(self) -> None:
        while self._running:
            try:
                await self._poll_once()
            except PermissionError:
                log.warning("Auth expired, re-logging in...")
                try:
                    await self._client.login(self._config.user, self._config.password)
                except Exception:
                    log.exception("Re-login failed")
            except Exception:
                log.exception("Poll error")

            await asyncio.sleep(self._config.poll_interval)

    async def _poll_once(self) -> None:
        # Poll monitored channels
        for name, rid in self._channel_ids.items():
            await self._check_room(rid, is_dm=False)

        # Poll DM rooms
        try:
            dms = await self._client.get_dm_rooms()
            for dm in dms:
                await self._check_room(dm["_id"], is_dm=True)
        except Exception:
            log.debug("Failed to list DM rooms", exc_info=True)

    async def _check_room(self, room_id: str, *, is_dm: bool) -> None:
        if self._processing.get(room_id):
            return

        oldest = self._last_ts.get(room_id, "")
        if not oldest:
            # First poll — set timestamp to now, don't process old messages
            from datetime import datetime, timezone
            self._last_ts[room_id] = datetime.now(timezone.utc).isoformat()
            return

        try:
            if is_dm:
                messages = await self._client.get_dm_history(room_id, oldest)
            else:
                messages = await self._client.get_history(room_id, oldest)
        except Exception:
            log.debug("Failed to get history for %s", room_id, exc_info=True)
            return

        if not messages:
            return

        # Messages come newest-first; reverse for chronological order
        messages.sort(key=lambda m: m.get("ts", ""))

        # Update last timestamp
        self._last_ts[room_id] = messages[-1]["ts"]

        # Filter out bot's own messages
        user_messages = [
            m for m in messages
            if m.get("u", {}).get("_id") != self._client.user_id
            and m.get("msg", "").strip()
        ]

        for msg in user_messages:
            text = msg["msg"].strip()
            asyncio.create_task(
                self._handle_message(room_id, text),
                name=f"rc-msg-{room_id}",
            )

    async def _handle_message(self, room_id: str, text: str) -> None:
        if self._processing.get(room_id):
            return
        self._processing[room_id] = True

        try:
            session_id = f"rc-{room_id}"
            result = await self._agent.run(text, session_id=session_id)
            reply = result.text or ""
            if reply.strip():
                await self._client.send_message(room_id, reply)
        except Exception:
            log.exception("Failed to handle message in room %s", room_id)
            try:
                await self._client.send_message(room_id, "처리 중 오류가 발생했습니다.")
            except Exception:
                pass
        finally:
            self._processing[room_id] = False
