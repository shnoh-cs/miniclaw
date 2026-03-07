"""OpenAI-compatible model provider for vLLM endpoints."""

from __future__ import annotations

import json
import re
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from openai import AsyncOpenAI

from openclaw.agent.types import (
    AgentMessage,
    ContentBlock,
    TextBlock,
    ThinkingLevel,
    TokenUsage,
    ToolDefinition,
    ToolUseBlock,
)
from openclaw.config import AppConfig, ModelOptionConfig


@dataclass
class StreamChunk:
    """A single chunk from the streaming response."""

    text: str = ""
    thinking: str = ""
    tool_calls: list[ToolUseBlock] = field(default_factory=list)
    finish_reason: str | None = None
    usage: TokenUsage | None = None


@dataclass
class ModelResponse:
    """Complete model response after stream is consumed."""

    message: AgentMessage | None = None
    usage: TokenUsage = field(default_factory=TokenUsage)
    finish_reason: str = ""
    raw_text: str = ""


class ModelProvider:
    """Wraps an OpenAI-compatible API (vLLM) with tool calling support."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.client = AsyncOpenAI(
            base_url=config.endpoints.llm.base_url,
            api_key=config.endpoints.llm.api_key,
        )

    def _get_model_options(self, model: str | None = None) -> ModelOptionConfig:
        model_id = model or self.config.models.default
        return self.config.models.options.get(model_id, ModelOptionConfig())

    def _get_tool_mode(self, model: str | None = None) -> str:
        return self._get_model_options(model).tool_mode

    async def stream(
        self,
        *,
        system: str,
        messages: list[AgentMessage],
        tools: list[ToolDefinition] | None = None,
        model: str | None = None,
        thinking: ThinkingLevel = ThinkingLevel.OFF,
    ) -> AsyncIterator[StreamChunk]:
        """Stream a response from the model.

        Automatically selects native or prompt-based tool calling.
        """
        model_id = model or self.config.models.default
        opts = self._get_model_options(model_id)
        tool_mode = opts.tool_mode

        # Decide tool calling strategy
        use_native = False
        if tools and tool_mode != "prompt":
            if tool_mode == "native":
                use_native = True
            else:  # auto
                use_native = True  # try native first

        # Build messages for the API
        api_messages = self._build_api_messages(system, messages, tools, use_native)

        # Build kwargs
        kwargs: dict[str, Any] = {
            "model": model_id,
            "messages": api_messages,
            "stream": True,
            "max_tokens": opts.max_tokens,
            "stream_options": {"include_usage": True},
        }

        if use_native and tools:
            kwargs["tools"] = [t.to_openai_schema() for t in tools]
            kwargs["tool_choice"] = "auto"

        if opts.stop_sequences:
            kwargs["stop"] = opts.stop_sequences
        elif not use_native and tools:
            kwargs["stop"] = ["</tool_call>"]

        try:
            stream = await self.client.chat.completions.create(**kwargs)
            async for chunk in stream:
                yield self._parse_chunk(chunk, use_native)
        except Exception as e:
            err_str = str(e)
            # If native tool calling failed, retry with prompt mode
            if use_native and tool_mode == "auto" and _is_tool_call_error(err_str):
                async for chunk in self._stream_prompt_mode(
                    model_id, api_messages, tools or [], opts
                ):
                    yield chunk
            else:
                raise

    async def _stream_prompt_mode(
        self,
        model_id: str,
        base_messages: list[dict[str, Any]],
        tools: list[ToolDefinition],
        opts: ModelOptionConfig,
    ) -> AsyncIterator[StreamChunk]:
        """Fallback: stream with prompt-based tool calling."""
        # Rebuild messages with tool descriptions in system prompt
        messages = _inject_tools_into_prompt(base_messages, tools)

        kwargs: dict[str, Any] = {
            "model": model_id,
            "messages": messages,
            "stream": True,
            "max_tokens": opts.max_tokens,
            "stop": ["</tool_call>"],
            "stream_options": {"include_usage": True},
        }

        stream = await self.client.chat.completions.create(**kwargs)
        async for chunk in stream:
            yield self._parse_chunk(chunk, native=False)

    async def complete(
        self,
        *,
        system: str,
        messages: list[AgentMessage],
        model: str | None = None,
        max_tokens: int = 4096,
    ) -> str:
        """Non-streaming completion (for compaction, memory flush, etc.)."""
        model_id = model or self.config.models.compaction_model
        api_messages = self._build_api_messages(system, messages, tools=None, native=False)

        response = await self.client.chat.completions.create(
            model=model_id,
            messages=api_messages,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content or ""

    def _build_api_messages(
        self,
        system: str,
        messages: list[AgentMessage],
        tools: list[ToolDefinition] | None,
        native: bool,
    ) -> list[dict[str, Any]]:
        """Convert internal messages to OpenAI API format."""
        api_msgs: list[dict[str, Any]] = []

        # System message
        system_text = system
        if not native and tools:
            system_text = _append_tool_descriptions(system, tools)
        api_msgs.append({"role": "system", "content": system_text})

        # Conversation messages
        for msg in messages:
            api_msg = self._convert_message(msg, native)
            if api_msg:
                api_msgs.append(api_msg)

        return api_msgs

    def _convert_message(
        self, msg: AgentMessage, native: bool
    ) -> dict[str, Any] | None:
        """Convert a single AgentMessage to API format."""
        if msg.role == "system":
            return {"role": "system", "content": msg.text}

        if msg.role == "user":
            # Check for tool results
            tool_results = msg.tool_results
            if tool_results:
                # In native mode, send as tool messages
                if native:
                    results = []
                    for tr in tool_results:
                        results.append({
                            "role": "tool",
                            "tool_call_id": tr.tool_use_id,
                            "content": tr.content,
                        })
                    return results[0] if len(results) == 1 else None  # handled below
                else:
                    # In prompt mode, include as text
                    parts = []
                    for tr in tool_results:
                        status = "ERROR" if tr.is_error else "OK"
                        parts.append(
                            f"<tool_result name=\"{tr.tool_use_id}\" status=\"{status}\">\n"
                            f"{tr.content}\n</tool_result>"
                        )
                    return {"role": "user", "content": "\n".join(parts)}

            # Regular user message
            content = msg.text
            if not content:
                return None
            return {"role": "user", "content": content}

        if msg.role == "assistant":
            tool_uses = msg.tool_uses
            if native and tool_uses:
                tool_calls = []
                for tu in tool_uses:
                    tool_calls.append({
                        "id": tu.id,
                        "type": "function",
                        "function": {
                            "name": tu.name,
                            "arguments": json.dumps(tu.input),
                        },
                    })
                return {
                    "role": "assistant",
                    "content": msg.text or None,
                    "tool_calls": tool_calls,
                }
            return {"role": "assistant", "content": msg.text}

        return None

    def _parse_chunk(self, chunk: Any, native: bool) -> StreamChunk:
        """Parse a streaming chunk into a StreamChunk."""
        result = StreamChunk()

        if not chunk.choices:
            # Usage-only chunk (last chunk in stream)
            if chunk.usage:
                result.usage = TokenUsage(
                    input_tokens=chunk.usage.prompt_tokens or 0,
                    output_tokens=chunk.usage.completion_tokens or 0,
                )
            return result

        choice = chunk.choices[0]
        delta = choice.delta

        if delta.content:
            result.text = delta.content

        if native and delta.tool_calls:
            for tc in delta.tool_calls:
                if tc.function and tc.function.name:
                    args = tc.function.arguments or "{}"
                    try:
                        parsed_args = json.loads(args)
                    except json.JSONDecodeError:
                        parsed_args = {}
                    result.tool_calls.append(
                        ToolUseBlock(
                            id=tc.id or f"toolu_{tc.index}",
                            name=tc.function.name,
                            input=parsed_args,
                        )
                    )

        result.finish_reason = choice.finish_reason
        return result


# ---------------------------------------------------------------------------
# Prompt-based tool calling helpers
# ---------------------------------------------------------------------------

TOOL_CALL_PATTERN = re.compile(
    r"<tool_call>\s*(\{.*?\})\s*(?:</tool_call>)?",
    re.DOTALL,
)


def parse_tool_calls_from_text(text: str) -> tuple[str, list[ToolUseBlock]]:
    """Extract tool calls from model text output (prompt-based mode).

    Returns (cleaned_text, tool_calls).
    """
    tool_calls: list[ToolUseBlock] = []
    cleaned = text

    for match in TOOL_CALL_PATTERN.finditer(text):
        try:
            data = json.loads(match.group(1))
            name = data.get("name", "")
            arguments = data.get("arguments", {})
            if isinstance(arguments, str):
                arguments = json.loads(arguments)
            tool_calls.append(
                ToolUseBlock(name=name, input=arguments)
            )
            cleaned = cleaned.replace(match.group(0), "")
        except (json.JSONDecodeError, KeyError):
            continue

    return cleaned.strip(), tool_calls


def _append_tool_descriptions(system: str, tools: list[ToolDefinition]) -> str:
    """Append tool descriptions to the system prompt for prompt-based mode."""
    if not tools:
        return system

    tool_section = "\n\n## Available Tools\n\n"
    tool_section += "You can call tools by wrapping a JSON object in <tool_call> tags:\n"
    tool_section += '<tool_call>\n{"name": "tool_name", "arguments": {"arg": "value"}}\n</tool_call>\n\n'
    tool_section += "Available tools:\n\n"

    for tool in tools:
        tool_section += tool.to_prompt_description() + "\n\n"

    return system + tool_section


def _inject_tools_into_prompt(
    messages: list[dict[str, Any]], tools: list[ToolDefinition]
) -> list[dict[str, Any]]:
    """Inject tool descriptions into the system message for prompt-based mode."""
    if not messages or not tools:
        return messages

    result = messages.copy()
    if result[0]["role"] == "system":
        result[0] = {
            "role": "system",
            "content": _append_tool_descriptions(result[0]["content"], tools),
        }
    return result


def _is_tool_call_error(error_str: str) -> bool:
    """Check if an error is related to tool calling not being supported."""
    indicators = [
        "tool",
        "function",
        "not supported",
        "invalid parameter",
        "unknown field",
    ]
    lower = error_str.lower()
    return any(ind in lower for ind in indicators)
