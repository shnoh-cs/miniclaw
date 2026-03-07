"""JSONL append-only session manager with write locking."""

from __future__ import annotations

import fcntl
import json
import os
import time
from pathlib import Path
from typing import Any

from openclaw.agent.types import (
    AgentMessage,
    CompactionEntry,
    ContentBlock,
    ImageBlock,
    SessionMeta,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
)


def _serialize_block(block: ContentBlock) -> dict[str, Any]:
    if isinstance(block, TextBlock):
        return {"type": "text", "text": block.text}
    if isinstance(block, ImageBlock):
        return {"type": "image", "source": block.source, "media_type": block.media_type}
    if isinstance(block, ToolUseBlock):
        return {"type": "tool_use", "id": block.id, "name": block.name, "input": block.input}
    if isinstance(block, ToolResultBlock):
        return {
            "type": "tool_result",
            "tool_use_id": block.tool_use_id,
            "content": block.content,
            "is_error": block.is_error,
        }
    return {}


def _deserialize_block(data: dict[str, Any]) -> ContentBlock:
    t = data.get("type", "text")
    if t == "text":
        return TextBlock(text=data.get("text", ""))
    if t == "image":
        return ImageBlock(source=data.get("source", ""), media_type=data.get("media_type", "image/png"))
    if t == "tool_use":
        return ToolUseBlock(
            id=data.get("id", ""),
            name=data.get("name", ""),
            input=data.get("input", {}),
        )
    if t == "tool_result":
        return ToolResultBlock(
            tool_use_id=data.get("tool_use_id", ""),
            content=data.get("content", ""),
            is_error=data.get("is_error", False),
        )
    return TextBlock(text=str(data))


class SessionWriteLock:
    """File-based mutex for concurrent session access."""

    def __init__(self, lock_path: Path, stale_seconds: int = 1800) -> None:
        self.lock_path = lock_path
        self.stale_seconds = stale_seconds
        self._fd: int | None = None

    def acquire(self) -> bool:
        """Acquire the write lock. Returns True if successful."""
        # Check for stale lock
        if self.lock_path.exists():
            try:
                content = self.lock_path.read_text().strip()
                if content:
                    parts = content.split(":")
                    if len(parts) == 2:
                        pid, ts = int(parts[0]), float(parts[1])
                        # Check if process is alive
                        try:
                            os.kill(pid, 0)
                        except OSError:
                            # Process is dead, remove stale lock
                            self.lock_path.unlink(missing_ok=True)
                        else:
                            # Process alive but lock might be stale
                            if time.time() - ts > self.stale_seconds:
                                self.lock_path.unlink(missing_ok=True)
                            else:
                                return False
            except (ValueError, OSError):
                self.lock_path.unlink(missing_ok=True)

        try:
            self.lock_path.parent.mkdir(parents=True, exist_ok=True)
            self._fd = os.open(
                str(self.lock_path),
                os.O_CREAT | os.O_EXCL | os.O_WRONLY,
            )
            os.write(self._fd, f"{os.getpid()}:{time.time()}".encode())
            fcntl.flock(self._fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return True
        except (OSError, IOError):
            if self._fd is not None:
                os.close(self._fd)
                self._fd = None
            return False

    def release(self) -> None:
        if self._fd is not None:
            try:
                fcntl.flock(self._fd, fcntl.LOCK_UN)
                os.close(self._fd)
            except OSError:
                pass
            finally:
                self._fd = None
                self.lock_path.unlink(missing_ok=True)

    def __enter__(self) -> SessionWriteLock:
        if not self.acquire():
            raise RuntimeError(f"Failed to acquire session lock: {self.lock_path}")
        return self

    def __exit__(self, *args: Any) -> None:
        self.release()


class SessionManager:
    """Manages a single session as an append-only JSONL file.

    Each line is either an AgentMessage or a CompactionEntry.
    """

    def __init__(self, session_dir: Path, session_id: str) -> None:
        self.session_dir = session_dir
        self.session_id = session_id
        self.file_path = session_dir / f"{session_id}.jsonl"
        self.lock_path = session_dir / f".{session_id}.lock"
        self.messages: list[AgentMessage] = []
        self.compaction_entries: list[CompactionEntry] = []
        self.meta = SessionMeta(session_id=session_id)
        self._loaded = False
        self._file_mtime: float = 0.0

    def load(self) -> None:
        """Load session from JSONL file."""
        self.messages.clear()
        self.compaction_entries.clear()

        if not self.file_path.exists():
            self._loaded = True
            return

        try:
            stat = self.file_path.stat()
            if stat.st_mtime == self._file_mtime and self._loaded:
                return  # cache hit
            self._file_mtime = stat.st_mtime
        except OSError:
            pass

        with open(self.file_path, encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue  # skip corrupt lines (repair)

                if data.get("type") == "compaction":
                    self.compaction_entries.append(CompactionEntry(**data))
                elif "role" in data:
                    blocks = [_deserialize_block(b) for b in data.get("content", [])]
                    msg = AgentMessage(
                        id=data.get("id", ""),
                        parent_id=data.get("parent_id"),
                        role=data["role"],
                        content=blocks,
                        timestamp=data.get("timestamp", 0.0),
                    )
                    self.messages.append(msg)

        self._loaded = True
        self._repair_tool_pairing()

    def append(self, message: AgentMessage) -> None:
        """Append a message to the session."""
        self.messages.append(message)
        self._write_entry(self._serialize_message(message))

    def append_compaction(self, entry: CompactionEntry) -> None:
        """Append a compaction entry and trim old messages."""
        self.compaction_entries.append(entry)
        self._write_entry(entry.model_dump())

    def replace_messages(self, messages: list[AgentMessage]) -> None:
        """Replace all messages (after compaction). Rewrites the file."""
        self.messages = messages
        self._rewrite()

    def get_messages_for_model(self, max_turns: int | None = None) -> list[AgentMessage]:
        """Get messages ready for the model, optionally limited."""
        msgs = list(self.messages)
        if max_turns and len(msgs) > max_turns * 2:
            # Keep last N turns (each turn = user + assistant)
            msgs = msgs[-(max_turns * 2):]
        return msgs

    @property
    def latest_compaction_summary(self) -> str | None:
        if self.compaction_entries:
            return self.compaction_entries[-1].summary
        return None

    def _serialize_message(self, msg: AgentMessage) -> dict[str, Any]:
        return {
            "id": msg.id,
            "parent_id": msg.parent_id,
            "role": msg.role,
            "content": [_serialize_block(b) for b in msg.content],
            "timestamp": msg.timestamp,
        }

    def _write_entry(self, data: dict[str, Any]) -> None:
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.file_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())

    def _rewrite(self) -> None:
        """Rewrite the entire session file (after compaction)."""
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.file_path.with_suffix(".tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            for entry in self.compaction_entries:
                f.write(json.dumps(entry.model_dump(), ensure_ascii=False) + "\n")
            for msg in self.messages:
                f.write(json.dumps(self._serialize_message(msg), ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
        tmp_path.replace(self.file_path)

    def _repair_tool_pairing(self) -> None:
        """Repair tool_use / tool_result pairing mismatches.

        Ensures every tool_use in an assistant message has a corresponding
        tool_result in the next user message.
        """
        if len(self.messages) < 2:
            return

        repaired = False
        for i in range(len(self.messages) - 1):
            msg = self.messages[i]
            if msg.role != "assistant":
                continue

            tool_uses = msg.tool_uses
            if not tool_uses:
                continue

            # Find next user message
            next_msg = self.messages[i + 1] if i + 1 < len(self.messages) else None
            if not next_msg or next_msg.role != "user":
                continue

            existing_ids = {tr.tool_use_id for tr in next_msg.tool_results}
            for tu in tool_uses:
                if tu.id not in existing_ids:
                    # Missing tool result — add a placeholder
                    next_msg.content.append(
                        ToolResultBlock(
                            tool_use_id=tu.id,
                            content="[Tool result missing — session repaired]",
                            is_error=True,
                        )
                    )
                    repaired = True

        if repaired:
            self._rewrite()

    def estimate_tokens(self) -> int:
        """Estimate total tokens using 4 chars per token heuristic."""
        total_chars = 0
        for msg in self.messages:
            for block in msg.content:
                if isinstance(block, TextBlock):
                    total_chars += len(block.text)
                elif isinstance(block, ToolUseBlock):
                    total_chars += len(json.dumps(block.input)) + len(block.name)
                elif isinstance(block, ToolResultBlock):
                    total_chars += len(block.content)
                elif isinstance(block, ImageBlock):
                    total_chars += 8000  # images ≈ 8000 chars
        return total_chars // 4
