"""Rocket.Chat integration — REST API polling bridge."""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

import httpx

from openclaw.config import RocketChatConfig

log = logging.getLogger("openclaw.rocketchat")


# ---------------------------------------------------------------------------
# Poll data structures
# ---------------------------------------------------------------------------


@dataclass
class PollTarget:
    """One user being polled."""

    username: str
    room_id: str


@dataclass
class ActivePoll:
    """Tracks a multi-user poll lifecycle."""

    poll_id: str
    question: str
    requester_name: str
    requester_room: str  # room_id to send aggregated results to
    targets: dict[str, PollTarget] = field(default_factory=dict)
    deadline: float = 0.0  # unix timestamp
    completed: bool = False


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

        # Poll tracking
        self._active_polls: dict[str, ActivePoll] = {}  # poll_id → poll
        self._poll_rooms: dict[str, str] = {}  # room_id → poll_id

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
        await self._client.send_message(room_id, message)
        # Ensure poller picks up this room and skips the message we just sent
        from datetime import datetime, timezone
        self._last_ts[room_id] = datetime.now(timezone.utc).isoformat()
        return room_id

    # ------------------------------------------------------------------
    # Poll system
    # ------------------------------------------------------------------

    async def create_poll(
        self,
        question: str,
        usernames: list[str],
        requester_name: str,
        requester_room: str,
        deadline_minutes: int = 60,
    ) -> ActivePoll:
        """Send *question* to each user via DM and track responses.

        Unlike the previous version, polled users have a natural conversation
        with the agent.  Responses are extracted from session history at the
        deadline rather than being blindly intercepted.
        """
        poll_id = uuid.uuid4().hex[:8]
        poll = ActivePoll(
            poll_id=poll_id,
            question=question,
            requester_name=requester_name,
            requester_room=requester_room,
            deadline=time.time() + deadline_minutes * 60,
        )

        failed: list[str] = []
        for username in usernames:
            try:
                # Create DM room and seed the agent session with poll context
                room_id = await self._client.create_dm(username)
                from datetime import datetime, timezone
                self._last_ts[room_id] = datetime.now(timezone.utc).isoformat()

                # Run through agent so it has full context for follow-ups
                prompt = (
                    f"[설문 진행] {requester_name}님을 대신하여 "
                    f"{username}님에게 설문을 전달해 주세요.\n\n"
                    f"질문: {question}\n\n"
                    f"자연스럽고 친근하게 전달하세요. "
                    f"유저가 추가 질문을 하면 아는 범위에서 답하고, "
                    f"모르면 {requester_name}님에게 직접 물어보라고 안내하세요."
                )
                result = await self._agent.run(
                    prompt, session_id=f"rc-{room_id}",
                )
                reply = result.text or ""
                if reply.strip():
                    await self._client.send_message(room_id, reply)
                    # Update timestamp past our own message
                    self._last_ts[room_id] = datetime.now(timezone.utc).isoformat()

                poll.targets[username] = PollTarget(username=username, room_id=room_id)
                self._poll_rooms[room_id] = poll_id
                log.info("Poll %s: sent to %s (room %s)", poll_id, username, room_id)
            except Exception:
                log.warning("Poll %s: failed to DM %s", poll_id, username, exc_info=True)
                failed.append(username)

        self._active_polls[poll_id] = poll

        # Schedule deadline aggregation
        delay = deadline_minutes * 60
        asyncio.create_task(
            self._poll_deadline(poll_id, delay), name=f"poll-deadline-{poll_id}"
        )

        if failed:
            log.warning("Poll %s: could not reach %s", poll_id, ", ".join(failed))

        return poll

    async def _poll_deadline(self, poll_id: str, delay: float) -> None:
        """Wait for deadline then aggregate responses from session history."""
        await asyncio.sleep(delay)
        poll = self._active_polls.get(poll_id)
        if poll and not poll.completed:
            log.info("Poll %s: deadline reached, aggregating", poll_id)
            await self._aggregate_poll(poll_id)

    async def _aggregate_poll(self, poll_id: str) -> None:
        """Aggregate poll responses from session history and send to requester."""
        poll = self._active_polls.get(poll_id)
        if not poll or poll.completed:
            return
        poll.completed = True

        # Clean up room mappings
        for target in poll.targets.values():
            self._poll_rooms.pop(target.room_id, None)

        # Build conversation summaries from each target's session
        summaries: list[str] = []
        for target in poll.targets.values():
            session_id = f"rc-{target.room_id}"
            session = self._agent._get_session(session_id)
            session.load()
            messages = session.messages()

            # Extract user messages (skip the initial poll question)
            user_msgs = [
                m.text_content() for m in messages
                if m.role == "user"
            ]
            if user_msgs:
                summaries.append(
                    f"- **{target.username}**: {' / '.join(user_msgs)}"
                )
            else:
                summaries.append(f"- **{target.username}**: (응답 없음)")

        summary_text = "\n".join(summaries)
        prompt = (
            f"설문이 마감되었습니다. 아래 응답들을 분석해서 결과를 정리하고, "
            f"최적의 일정을 추천해 주세요.\n\n"
            f"질문: {poll.question}\n"
            f"요청자: {poll.requester_name}\n\n"
            f"응답:\n{summary_text}"
        )

        try:
            result = await self._agent.run(
                prompt, session_id=f"rc-{poll.requester_room}",
            )
            reply = result.text or summary_text
            await self._client.send_message(poll.requester_room, reply)
        except Exception:
            log.exception("Poll %s: aggregation failed, sending raw summary", poll_id)
            try:
                await self._client.send_message(
                    poll.requester_room,
                    f"**설문 결과** (질문: {poll.question})\n\n{summary_text}",
                )
            except Exception:
                pass

        self._active_polls.pop(poll_id, None)

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
