"""Core data types for the OpenClaw agent harness."""

from __future__ import annotations

import time
import uuid
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Content blocks (modeled after Anthropic message format)
# ---------------------------------------------------------------------------


class TextBlock(BaseModel):
    type: Literal["text"] = "text"
    text: str


class ImageBlock(BaseModel):
    type: Literal["image"] = "image"
    source: str  # file path or URL
    media_type: str = "image/png"


class ToolUseBlock(BaseModel):
    type: Literal["tool_use"] = "tool_use"
    id: str = Field(default_factory=lambda: f"toolu_{uuid.uuid4().hex[:24]}")
    name: str
    input: dict[str, Any] = Field(default_factory=dict)


class ToolResultBlock(BaseModel):
    type: Literal["tool_result"] = "tool_result"
    tool_use_id: str
    content: str = ""
    is_error: bool = False


ContentBlock = TextBlock | ImageBlock | ToolUseBlock | ToolResultBlock


# ---------------------------------------------------------------------------
# Messages
# ---------------------------------------------------------------------------


class AgentMessage(BaseModel):
    """A single turn in the conversation."""

    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    parent_id: str | None = None
    role: Literal["user", "assistant", "system"]
    content: list[ContentBlock] = Field(default_factory=list)
    timestamp: float = Field(default_factory=time.time)

    @property
    def text(self) -> str:
        """Extract concatenated text from all text blocks."""
        return "\n".join(b.text for b in self.content if isinstance(b, TextBlock))

    @property
    def tool_uses(self) -> list[ToolUseBlock]:
        return [b for b in self.content if isinstance(b, ToolUseBlock)]

    @property
    def tool_results(self) -> list[ToolResultBlock]:
        return [b for b in self.content if isinstance(b, ToolResultBlock)]


# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------


class ToolParameter(BaseModel):
    name: str
    type: str = "string"
    description: str = ""
    required: bool = True
    enum: list[str] | None = None


class ToolDefinition(BaseModel):
    """Schema for a tool that the agent can call."""

    name: str
    description: str
    parameters: list[ToolParameter] = Field(default_factory=list)

    def to_openai_schema(self) -> dict[str, Any]:
        """Convert to OpenAI function-calling format."""
        properties: dict[str, Any] = {}
        required: list[str] = []
        for p in self.parameters:
            prop: dict[str, Any] = {"type": p.type, "description": p.description}
            if p.enum:
                prop["enum"] = p.enum
            properties[p.name] = prop
            if p.required:
                required.append(p.name)
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        }

    def to_prompt_description(self) -> str:
        """Convert to a text description for prompt-based tool calling."""
        params_desc = []
        for p in self.parameters:
            req = " (required)" if p.required else " (optional)"
            params_desc.append(f"  - {p.name}: {p.type}{req} — {p.description}")
        params_text = "\n".join(params_desc) if params_desc else "  (no parameters)"
        return f"### {self.name}\n{self.description}\nParameters:\n{params_text}"


# ---------------------------------------------------------------------------
# Tool result (runtime)
# ---------------------------------------------------------------------------


class ToolResult(BaseModel):
    """Result of executing a tool."""

    tool_use_id: str
    content: str = ""
    is_error: bool = False


# ---------------------------------------------------------------------------
# Thinking levels
# ---------------------------------------------------------------------------


class ThinkingLevel(str, Enum):
    OFF = "off"
    MINIMAL = "minimal"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    XHIGH = "xhigh"
    ADAPTIVE = "adaptive"

    @classmethod
    def from_str(cls, value: str) -> ThinkingLevel:
        aliases = {
            "think": cls.MINIMAL,
            "think hard": cls.LOW,
            "think harder": cls.MEDIUM,
            "ultrathink": cls.HIGH,
            "ultrathink+": cls.XHIGH,
            "x-high": cls.XHIGH,
            "extra-high": cls.XHIGH,
        }
        normalized = value.strip().lower()
        if normalized in aliases:
            return aliases[normalized]
        try:
            return cls(normalized)
        except ValueError:
            return cls.OFF

    def fallback(self) -> ThinkingLevel:
        """Return the next lower thinking level."""
        order = [
            ThinkingLevel.XHIGH,
            ThinkingLevel.HIGH,
            ThinkingLevel.MEDIUM,
            ThinkingLevel.LOW,
            ThinkingLevel.MINIMAL,
            ThinkingLevel.OFF,
        ]
        try:
            idx = order.index(self)
            return order[min(idx + 1, len(order) - 1)]
        except ValueError:
            return ThinkingLevel.OFF


# ---------------------------------------------------------------------------
# Compaction entry
# ---------------------------------------------------------------------------


class CompactionEntry(BaseModel):
    """A compaction summary stored in the session JSONL."""

    type: Literal["compaction"] = "compaction"
    summary: str
    tokens_before: int
    tokens_after: int
    first_kept_entry_id: str | None = None
    timestamp: float = Field(default_factory=time.time)


# ---------------------------------------------------------------------------
# Session metadata
# ---------------------------------------------------------------------------


class SessionMeta(BaseModel):
    session_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    session_key: str = ""
    agent_id: str = "default"
    created_at: float = Field(default_factory=time.time)
    model: str = ""
    workspace: str = ""


# ---------------------------------------------------------------------------
# Token usage
# ---------------------------------------------------------------------------


class TokenUsage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0

    @property
    def total(self) -> int:
        return self.input_tokens + self.output_tokens


# ---------------------------------------------------------------------------
# Run result
# ---------------------------------------------------------------------------


class RunResult(BaseModel):
    """Result of a single agent run."""

    text: str = ""
    messages: list[AgentMessage] = Field(default_factory=list)
    usage: TokenUsage = Field(default_factory=TokenUsage)
    tool_calls_count: int = 0
    compacted: bool = False
    error: str | None = None

    @property
    def success(self) -> bool:
        return self.error is None


# ---------------------------------------------------------------------------
# Error classification (for failover)
# ---------------------------------------------------------------------------


class FailoverReason(str, Enum):
    AUTH = "auth"
    AUTH_PERMANENT = "auth_permanent"
    BILLING = "billing"
    RATE_LIMIT = "rate_limit"
    OVERLOADED = "overloaded"
    TIMEOUT = "timeout"
    CONTEXT_OVERFLOW = "context_overflow"
    MODEL_NOT_FOUND = "model_not_found"
    FORMAT = "format"
    SESSION_EXPIRED = "session_expired"
    TRANSIENT = "transient"
    UNKNOWN = "unknown"
