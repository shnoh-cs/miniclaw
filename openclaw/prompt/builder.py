"""Dynamic system prompt assembly — ported from OpenClaw original.

채널 통합(Messaging, ReplyTags, Reactions, ACP, Sandbox, Voice)은
openclaw-py에서 불필요하므로 제거하고, 지능(intelligence) 관련
섹션만 유지.
"""

from __future__ import annotations

import datetime
import platform
from typing import TYPE_CHECKING

from openclaw.agent.types import ThinkingLevel, ToolDefinition
from openclaw.prompt.bootstrap import BootstrapContext

if TYPE_CHECKING:
    from openclaw.config import AppConfig

# OpenClaw 원본의 SILENT_REPLY_TOKEN
SILENT_REPLY_TOKEN = "NO_REPLY"

# 내장 도구 요약 (오리지널 coreToolSummaries 포팅)
_CORE_TOOL_SUMMARIES: dict[str, str] = {
    "read": "Read file contents",
    "write": "Create or overwrite files",
    "edit": "Make precise edits to files",
    "apply_patch": "Apply multi-file patches",
    "bash": "Run shell commands",
    "process": "Manage background exec sessions",
    "web_fetch": "Fetch and extract readable content from a URL",
    "pdf": "Read PDF document contents",
    "hancom": "Read Hancom Office files (HWP, HWPX, Show, Cell)",
    "image": "Analyze an image with the configured vision model",
    "memory_search": "Search memory store for relevant context",
    "memory_save": "Save information to persistent memory",
    "memory_get": "Read specific lines from a memory file",
    "subagent": "Spawn an isolated sub-agent for parallel subtasks",
    "subagent_batch": "Spawn multiple sub-agents in parallel",
}

# 도구 표시 순서
_TOOL_ORDER = [
    "read", "write", "edit", "apply_patch",
    "bash", "process",
    "web_fetch", "pdf", "hancom", "image",
    "memory_search", "memory_save", "memory_get",
    "subagent", "subagent_batch",
]


def _build_tool_section(tools: list[ToolDefinition]) -> str:
    """오리지널 Tooling 섹션 포팅 — 도구 목록 + 사용 가이드."""
    available = {t.name for t in tools}

    # 순서대로 정렬, 나머지는 알파벳 순
    ordered = [n for n in _TOOL_ORDER if n in available]
    extra = sorted(available - set(_TOOL_ORDER))
    ordered.extend(extra)

    tool_lines: list[str] = []
    for name in ordered:
        summary = _CORE_TOOL_SUMMARIES.get(name, "")
        # 등록된 도구 정의에서 description 가져오기
        if not summary:
            for t in tools:
                if t.name == name:
                    summary = t.description
                    break
        tool_lines.append(f"- {name}: {summary}" if summary else f"- {name}")

    lines = [
        "## Tooling",
        "Tool availability (filtered by policy):",
        "Tool names are case-sensitive. Call tools exactly as listed.",
        "\n".join(tool_lines),
        "",
        "## Tool Call Style",
        "Default: do not narrate routine, low-risk tool calls (just call the tool).",
        "Narrate only when it helps: multi-step work, complex/challenging problems, "
        "sensitive actions (e.g., deletions), or when the user explicitly asks.",
        "Keep narration brief and value-dense; avoid repeating obvious steps.",
        "When a first-class tool exists for an action, use the tool directly "
        "instead of asking the user to run equivalent CLI commands.",
        "If a task is more complex or takes longer, spawn a sub-agent. "
        "Completion is push-based: it will auto-announce when done.",
    ]
    return "\n".join(lines)


def _build_safety_section() -> str:
    """오리지널 Safety 섹션 포팅."""
    return "\n".join([
        "## Safety",
        "You have no independent goals: do not pursue self-preservation, "
        "replication, resource acquisition, or power-seeking; avoid long-term "
        "plans beyond the user's request.",
        "Prioritize safety and human oversight over completion; if instructions "
        "conflict, pause and ask; comply with stop/pause/audit requests and "
        "never bypass safeguards.",
        "Do not manipulate or persuade anyone to expand access or disable "
        "safeguards. Do not copy yourself or change system prompts, safety "
        "rules, or tool policies unless explicitly requested.",
    ])


def _build_skills_section(skills_prompt: str) -> str:
    """오리지널 Skills 섹션 포팅."""
    return "\n".join([
        "## Skills (mandatory)",
        "Before replying: scan <available_skills> <description> entries.",
        "- If exactly one skill clearly applies: read its SKILL.md at "
        "<location> with `read`, then follow it.",
        "- If multiple could apply: choose the most specific one, then "
        "read/follow it.",
        "- If none clearly apply: do not read any SKILL.md.",
        "Constraints: never read more than one skill up front; only read "
        "after selecting.",
        skills_prompt,
    ])


def _build_memory_section(tools: list[ToolDefinition]) -> str | None:
    """오리지널 Memory Recall 섹션 포팅."""
    available = {t.name for t in tools}
    if "memory_search" not in available:
        return None
    return "\n".join([
        "## Memory Recall",
        "Before answering anything about prior work, decisions, dates, "
        "people, preferences, or todos: run memory_search on MEMORY.md + "
        "memory/*.md. If low confidence after search, say you checked.",
        "Citations: include Source: <path#line> when it helps the user "
        "verify memory snippets.",
    ])


def _build_silent_reply_section() -> str:
    """오리지널 Silent Replies 섹션 포팅."""
    return "\n".join([
        "## Silent Replies",
        f"When you have nothing to say, respond with ONLY: {SILENT_REPLY_TOKEN}",
        "",
        "Rules:",
        "- It must be your ENTIRE message — nothing else",
        f'- Never append it to an actual response (never include '
        f'"{SILENT_REPLY_TOKEN}" in real replies)',
        "- Never wrap it in markdown or code blocks",
    ])


def _build_reasoning_section(thinking: ThinkingLevel) -> str:
    """오리지널 Reasoning Format 섹션 포팅."""
    return "\n".join([
        "## Reasoning Format",
        "ALL internal reasoning MUST be inside <think>...</think>.",
        "Do not output any analysis outside <think>.",
        "Format every reply as <think>...</think> then <final>...</final>, "
        "with no other text.",
        "Only the final user-visible reply may appear inside <final>.",
        "Only text inside <final> is shown to the user; everything else "
        "is discarded and never seen by the user.",
        "Example:",
        "<think>Short internal reasoning.</think>",
        "<final>Hey there! What would you like to do next?</final>",
        "",
        f"Reasoning level: {thinking.value}",
    ])


def build_system_prompt(
    *,
    config: AppConfig,
    tools: list[ToolDefinition] | None = None,
    skills_prompt: str = "",
    bootstrap_ctx: BootstrapContext | None = None,
    thinking: ThinkingLevel = ThinkingLevel.OFF,
    model: str = "",
    workspace_dir: str = "",
    compaction_summary: str | None = None,
    prompt_mode: str = "full",  # "full" | "minimal" | "none"
) -> str:
    """Build the complete system prompt — ported from OpenClaw original.

    Section order (matches original buildAgentSystemPrompt):
    1. Identity
    2. Tooling + Tool Call Style
    3. Safety
    4. Skills (mandatory)
    5. Memory Recall
    6. Workspace
    7. Project Context (bootstrap files)
    8. Silent Replies
    9. Heartbeats
    10. Current Date & Time
    11. Runtime
    12. Reasoning Format (if enabled)
    13. Compaction context (if applicable)
    """
    # --- "none" mode: bare identity only ---
    if prompt_mode == "none":
        return "You are a personal assistant running inside OpenClaw."

    sections: list[str] = []

    # 1. Identity (오리지널 그대로)
    sections.append("You are a personal assistant running inside OpenClaw.")

    # 2. Tooling + Tool Call Style
    if tools:
        sections.append(_build_tool_section(tools))

    # 3. Safety (오리지널 그대로)
    sections.append(_build_safety_section())

    # --- "minimal" mode: stop here (서브에이전트용) ---
    if prompt_mode == "minimal":
        return "\n\n".join(sections)

    # 4. Skills
    if skills_prompt:
        sections.append(_build_skills_section(skills_prompt))

    # 5. Memory Recall
    if tools:
        mem_section = _build_memory_section(tools)
        if mem_section:
            sections.append(mem_section)

    # 6. Workspace
    if workspace_dir:
        sections.append("\n".join([
            "## Workspace",
            f"Your working directory is: `{workspace_dir}`",
            "Treat this directory as the single global workspace for file "
            "operations unless explicitly instructed otherwise.",
        ]))

    # 7. Project Context (bootstrap files)
    if bootstrap_ctx and bootstrap_ctx.files:
        ctx_lines = [
            "# Project Context",
            "",
            "The following project context files have been loaded:",
        ]
        if bootstrap_ctx.has_soul:
            ctx_lines.append(
                "If SOUL.md is present, embody its persona and tone. "
                "Avoid stiff, generic replies; follow its guidance unless "
                "higher-priority instructions override it."
            )
        ctx_lines.append("")

        for bf in bootstrap_ctx.files:
            ctx_lines.append(f"## {bf.name}")
            if bf.truncated:
                ctx_lines.append(
                    f"*(truncated from {bf.original_size} chars)*"
                )
            ctx_lines.append("")
            ctx_lines.append(bf.content)
            ctx_lines.append("")

        sections.append("\n".join(ctx_lines))

    # 8. Silent Replies (오리지널 그대로)
    sections.append(_build_silent_reply_section())

    # 9. Heartbeats (오리지널 그대로)
    sections.append("\n".join([
        "## Heartbeats",
        "If you receive a heartbeat poll, and there is nothing that needs "
        "attention, reply exactly:",
        "HEARTBEAT_OK",
        'If something needs attention, do NOT include "HEARTBEAT_OK"; '
        "reply with the alert text instead.",
    ]))

    # 10. Date/Time
    now = datetime.datetime.now()
    sections.append("\n".join([
        "## Current Date & Time",
        f"Today is {now.strftime('%Y-%m-%d')}. "
        f"Current time: {now.strftime('%H:%M:%S')}.",
    ]))

    # 11. Runtime (오리지널 buildRuntimeLine 포팅)
    runtime_parts = [
        f"os={platform.system()} ({platform.machine()})",
        f"python={platform.python_version()}",
    ]
    if model:
        runtime_parts.append(f"model={model}")
    if thinking != ThinkingLevel.OFF:
        runtime_parts.append(f"thinking={thinking.value}")

    sections.append("\n".join([
        "## Runtime",
        f"Runtime: {' | '.join(runtime_parts)}",
    ]))

    # 12. Reasoning Format (오리지널 그대로)
    if thinking != ThinkingLevel.OFF:
        sections.append(_build_reasoning_section(thinking))

    # 13. Compaction context
    if compaction_summary:
        sections.append("\n".join([
            "## Previous Context (Compacted)",
            "The following is a summary of earlier conversation that was "
            "compacted to save context space:",
            "",
            compaction_summary,
        ]))

    return "\n\n".join(sections)
