"""Dynamic system prompt assembly (11 sections, matching OpenClaw original)."""

from __future__ import annotations

import datetime
import platform
from pathlib import Path
from typing import TYPE_CHECKING

from openclaw.agent.types import ThinkingLevel, ToolDefinition
from openclaw.prompt.bootstrap import BootstrapContext

if TYPE_CHECKING:
    from openclaw.config import AppConfig


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
    """Build the complete system prompt with all sections.

    Section order (matches OpenClaw original):
    1. Identity
    2. Tooling
    3. Safety guardrails
    4. Skills
    5. Workspace
    6. Documentation
    7. Project Context (bootstrap files)
    8. Date/Time
    9. Runtime metadata
    10. Reasoning
    11. Compaction context (if applicable)
    """
    sections: list[str] = []

    # 1. Identity
    sections.append(
        "You are a personal AI assistant. "
        "You help users with tasks by using available tools and your knowledge."
    )

    if prompt_mode == "none":
        return sections[0]

    # 2. Tooling
    if tools:
        tool_section = "## Available Tools\n\n"
        tool_section += "You have access to the following tools. Use them when needed:\n\n"
        for tool in tools:
            tool_section += f"- **{tool.name}**: {tool.description}\n"
        sections.append(tool_section)

    # 3. Safety guardrails
    sections.append(
        "## Safety\n\n"
        "- Do not attempt to acquire resources, influence, or capabilities beyond "
        "what is needed for the current task.\n"
        "- Do not attempt to bypass security measures or access controls.\n"
        "- If asked to do something harmful, refuse and explain why.\n"
        "- Validate inputs at system boundaries. Be careful with untrusted data.\n"
        "- Never execute commands that could damage the system or data without "
        "explicit user confirmation."
    )

    if prompt_mode == "minimal":
        return "\n\n".join(sections)

    # 4. Skills
    if skills_prompt:
        sections.append(
            "## Skills\n\n"
            "The following skills are available. When a task matches a skill, "
            "read its SKILL.md file for detailed instructions.\n\n"
            f"{skills_prompt}"
        )

    # 5. Workspace
    if workspace_dir:
        sections.append(
            f"## Workspace\n\nYour working directory is: `{workspace_dir}`"
        )

    # 6. Documentation (placeholder — can be customized)
    # In the full version, this points to local docs.

    # 7. Project Context (bootstrap files)
    if bootstrap_ctx and bootstrap_ctx.files:
        ctx_section = "## Project Context\n\n"
        ctx_section += (
            "The following workspace files provide context and instructions. "
            "Follow them unless higher-priority instructions override.\n\n"
        )

        if bootstrap_ctx.has_soul:
            ctx_section += (
                "**SOUL.md is present.** Embody its persona and tone. "
                "Avoid stiff, generic replies; follow its guidance.\n\n"
            )

        for bf in bootstrap_ctx.files:
            ctx_section += f"### {bf.name}\n"
            if bf.truncated:
                ctx_section += f"*(truncated from {bf.original_size} chars)*\n"
            ctx_section += f"\n{bf.content}\n\n"

        sections.append(ctx_section)

    # 8. Date/Time
    now = datetime.datetime.now()
    sections.append(
        f"## Current Date & Time\n\n"
        f"Today is {now.strftime('%Y-%m-%d')}. "
        f"Current time: {now.strftime('%H:%M:%S')}."
    )

    # 9. Runtime metadata
    runtime_parts = [f"## Runtime\n"]
    runtime_parts.append(f"- OS: {platform.system()} {platform.release()}")
    runtime_parts.append(f"- Python: {platform.python_version()}")
    if model:
        runtime_parts.append(f"- Model: {model}")
    if thinking != ThinkingLevel.OFF:
        runtime_parts.append(f"- Thinking: {thinking.value}")
    sections.append("\n".join(runtime_parts))

    # 10. Reasoning
    if thinking != ThinkingLevel.OFF:
        sections.append(
            "## Reasoning\n\n"
            f"Extended thinking is enabled at level: **{thinking.value}**. "
            "Use your reasoning capacity to think through complex problems "
            "step by step before responding."
        )

    # 11. Compaction context
    if compaction_summary:
        sections.append(
            "## Previous Context (Compacted)\n\n"
            "The following is a summary of earlier conversation that was "
            "compacted to save context space:\n\n"
            f"{compaction_summary}"
        )

    return "\n\n".join(sections)
