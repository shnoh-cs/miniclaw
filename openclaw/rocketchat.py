"""Rocket.Chat integration — REST API polling bridge."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import httpx

from openclaw.config import RocketChatConfig

log = logging.getLogger("openclaw.rocketchat")


# ---------------------------------------------------------------------------
# Rocket.Chat REST client
# ---------------------------------------------------------------------------


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

    async def create_dm(self, username: str) -> str:
        """Create (or get existing) DM channel with *username*, return room_id."""
        resp = await self._client.post(
            "/api/v1/im.create", json={"username": username}
        )
        resp.raise_for_status()
        return resp.json()["room"]["_id"]

    async def close(self) -> None:
        await self._client.aclose()


# ---------------------------------------------------------------------------
# Polling bridge
# ---------------------------------------------------------------------------


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
        # room_id → username (populated from incoming messages)
        self._room_usernames: dict[str, str] = {}
        # Subagent DM session linking
        self._dm_session_map: dict[str, str] = {}   # room_id → subagent session_key
        self._dm_parent_session: dict[str, str] = {}  # room_id → parent session_id
        self._dm_task: dict[str, str] = {}  # room_id → subagent task description
        # Per-session locks to prevent concurrent agent.run() on same session
        self._session_locks: dict[str, asyncio.Lock] = {}

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

    def get_username_for_room(self, room_id: str) -> str:
        """Get the username associated with a DM room."""
        return self._room_usernames.get(room_id, "")

    # ------------------------------------------------------------------
    # Notifications
    # ------------------------------------------------------------------

    async def send_notification(self, text: str, reply_to: str = "") -> None:
        """Send a cron notification to the originating room or fallback channel."""
        if reply_to.startswith("rc-"):
            room_id = reply_to[3:]
            try:
                await self._client.send_message(room_id, text)
                return
            except Exception:
                log.warning("Failed to send cron reply to room %s", room_id)

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

    # ------------------------------------------------------------------
    # Send DM (proactive)
    # ------------------------------------------------------------------

    async def send_dm(self, username: str, message: str) -> str:
        """Create/get DM with *username* and send *message*. Returns room_id."""
        room_id = await self._client.create_dm(username)
        from datetime import datetime, timezone

        await self._client.send_message(room_id, message)
        self._last_ts[room_id] = datetime.now(timezone.utc).isoformat()
        return room_id

    def link_dm_session(
        self, room_id: str, session_key: str, parent_session: str,
        task: str = "",
    ) -> None:
        """Link a DM room to a subagent session.

        Replies in this room will be routed to *session_key* instead of
        the default ``rc-{room_id}`` session, and results will be announced
        back to *parent_session*.
        """
        self._dm_session_map[room_id] = session_key
        self._dm_parent_session[room_id] = parent_session
        self._dm_task[room_id] = task
        log.info(
            "DM room %s linked: session=%s, parent=%s",
            room_id, session_key, parent_session,
        )

    def _get_session_lock(self, session_id: str) -> asyncio.Lock:
        if session_id not in self._session_locks:
            self._session_locks[session_id] = asyncio.Lock()
        return self._session_locks[session_id]

    # ------------------------------------------------------------------
    # Main polling loop
    # ------------------------------------------------------------------

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

        # Filter out bot's own messages and track usernames
        user_messages = []
        for m in messages:
            sender = m.get("u", {})
            if sender.get("_id") == self._client.user_id:
                continue
            msg_text = m.get("msg", "").strip()
            if not msg_text:
                continue
            username = sender.get("username", "")
            if username:
                self._room_usernames[room_id] = username
            user_messages.append(m)

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
            parent_session = self._dm_parent_session.get(room_id)

            if parent_session:
                # Linked DM room (subagent): don't run agent.run() on
                # the subagent session — just ack and forward to parent.
                # This avoids cascading agent calls that produce multiple messages.
                log.info(
                    "Linked DM reply in room %s → forwarding to parent %s: %s",
                    room_id, parent_session, text[:100],
                )
                await self._client.send_message(room_id, "답변 감사합니다!")
                username = self._room_usernames.get(room_id, "unknown")
                await self._announce_to_parent(
                    parent_session, username, text,
                )
            else:
                # Normal room: run through agent as usual
                log.info("Normal message in room %s: %s", room_id, text[:100])
                session_id = f"rc-{room_id}"
                async with self._get_session_lock(session_id):
                    result = await self._agent.run(
                        text, session_id=session_id,
                    )
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

    async def _announce_to_parent(
        self, parent_session: str, username: str, reply_text: str,
    ) -> None:
        """Forward a DM reply to the parent session as a notification."""
        announce = (
            f"[서브에이전트 알림] {username}님이 답변했습니다: {reply_text}\n"
            f"이 알림에 대해 간단히 확인만 해주세요. 추가 작업(서브에이전트 생성, "
            f"DM 전송 등)은 하지 마세요."
        )
        log.info("Announcing to parent %s: %s replied '%s'",
                 parent_session, username, reply_text[:100])
        try:
            async with self._get_session_lock(parent_session):
                result = await self._agent.run(announce, session_id=parent_session)
            reply = result.text or ""
            log.info("Parent response: %s", reply[:200] if reply else "(empty)")
            # Send the agent's response to the parent's room
            if parent_session.startswith("rc-") and reply.strip():
                parent_room = parent_session[3:]
                await self._client.send_message(parent_room, reply)
        except Exception:
            log.exception(
                "Failed to announce %s's reply to parent session %s",
                username, parent_session,
            )
