"""Skills discovery, loading, and prompt generation with multi-level precedence."""

from __future__ import annotations

import os
import platform
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class SkillEntry:
    """A loaded skill with metadata from SKILL.md frontmatter."""

    name: str
    description: str = ""
    path: Path = Path()
    source: str = ""  # bundled, managed, personal, project, workspace
    user_invocable: bool = True
    disable_model_invocation: bool = False
    homepage: str = ""
    os_filter: list[str] = field(default_factory=list)
    required_bins: list[str] = field(default_factory=list)
    any_bins: list[str] = field(default_factory=list)
    required_env: list[str] = field(default_factory=list)
    env_overrides: dict[str, str] = field(default_factory=dict)

    @property
    def skill_md_path(self) -> Path:
        return self.path / "SKILL.md"


@dataclass
class SkillsSnapshot:
    """Snapshot of all eligible skills for a session."""

    skills: list[SkillEntry] = field(default_factory=list)
    total_chars: int = 0


def _parse_skill_md(skill_dir: Path) -> SkillEntry | None:
    """Parse SKILL.md frontmatter to extract skill metadata."""
    skill_md = skill_dir / "SKILL.md"
    if not skill_md.is_file():
        return None

    try:
        content = skill_md.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None

    # Parse YAML frontmatter (between --- markers)
    if not content.startswith("---"):
        return SkillEntry(name=skill_dir.name, path=skill_dir)

    end_idx = content.find("---", 3)
    if end_idx < 0:
        return SkillEntry(name=skill_dir.name, path=skill_dir)

    frontmatter_text = content[3:end_idx].strip()
    try:
        fm: dict[str, Any] = yaml.safe_load(frontmatter_text) or {}
    except yaml.YAMLError:
        return SkillEntry(name=skill_dir.name, path=skill_dir)

    # Extract openclaw-specific metadata
    oc_meta = {}
    if isinstance(fm.get("metadata"), dict):
        oc_meta = fm["metadata"].get("openclaw", {})
    if not isinstance(oc_meta, dict):
        oc_meta = {}

    return SkillEntry(
        name=fm.get("name", skill_dir.name),
        description=fm.get("description", ""),
        path=skill_dir,
        user_invocable=fm.get("user-invocable", True),
        disable_model_invocation=fm.get("disable-model-invocation", False),
        homepage=oc_meta.get("homepage", ""),
        os_filter=oc_meta.get("os", []),
        required_bins=oc_meta.get("requires", {}).get("bins", []),
        any_bins=oc_meta.get("requires", {}).get("anyBins", []),
        required_env=oc_meta.get("requires", {}).get("env", []),
    )


def _is_eligible(entry: SkillEntry) -> bool:
    """Check if a skill is eligible on the current platform."""
    # OS filter
    if entry.os_filter:
        current_os = platform.system().lower()
        os_map = {"darwin": "darwin", "linux": "linux", "windows": "win32"}
        if os_map.get(current_os, current_os) not in entry.os_filter:
            return False

    # Required binaries
    for bin_name in entry.required_bins:
        if not shutil.which(bin_name):
            return False

    # Any of these binaries
    if entry.any_bins:
        if not any(shutil.which(b) for b in entry.any_bins):
            return False

    # Required environment variables
    for env_var in entry.required_env:
        if not os.environ.get(env_var):
            return False

    return True


def load_skills(
    skill_dirs: list[Path],
    max_skills: int = 150,
) -> SkillsSnapshot:
    """Load skills from multiple directories with precedence.

    Later directories in the list have higher precedence.
    Precedence: bundled → managed → personal → project → workspace
    """
    skills_by_name: dict[str, SkillEntry] = {}

    for skill_dir in skill_dirs:
        if not skill_dir.is_dir():
            continue

        for entry_dir in sorted(skill_dir.iterdir()):
            if not entry_dir.is_dir():
                continue
            if entry_dir.name.startswith("."):
                continue

            entry = _parse_skill_md(entry_dir)
            if entry is None:
                continue

            entry.source = skill_dir.name
            if not _is_eligible(entry):
                continue

            # Higher precedence overrides lower
            skills_by_name[entry.name] = entry

    # Filter out model-disabled skills
    eligible = [
        s for s in skills_by_name.values()
        if not s.disable_model_invocation
    ]

    # Limit
    if len(eligible) > max_skills:
        eligible = eligible[:max_skills]

    total_chars = sum(
        97 + len(s.name) + len(s.description) + len(str(s.path))
        for s in eligible
    )

    return SkillsSnapshot(skills=eligible, total_chars=total_chars)


def build_skills_prompt(
    snapshot: SkillsSnapshot,
    max_chars: int = 30000,
) -> str:
    """Format skills snapshot into a prompt section."""
    if not snapshot.skills:
        return ""

    lines = ["<available_skills>"]
    total = 0

    for skill in snapshot.skills:
        entry_line = f"- **{skill.name}**: {skill.description}"
        if skill.path:
            entry_line += f" (read `{skill.skill_md_path}` for instructions)"

        if total + len(entry_line) > max_chars:
            lines.append(f"\n... ({len(snapshot.skills) - len(lines) + 1} more skills, truncated)")
            break

        lines.append(entry_line)
        total += len(entry_line)

    lines.append("</available_skills>")
    lines.append(
        "\nScan the available skills above. When a task matches a skill, "
        "use the `read` tool to load its SKILL.md for detailed instructions."
    )

    return "\n".join(lines)
