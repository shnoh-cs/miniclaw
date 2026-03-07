"""Bootstrap file loading: 8 standard files with budget management and truncation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from openclaw.config import BootstrapConfig

# Standard bootstrap filenames recognized by the system
BOOTSTRAP_FILENAMES = [
    "AGENTS.md",
    "SOUL.md",
    "TOOLS.md",
    "IDENTITY.md",
    "USER.md",
    "HEARTBEAT.md",
    "BOOTSTRAP.md",
    "MEMORY.md",
]

# Sub-agents only receive these files
SUBAGENT_ALLOWED = {"AGENTS.md", "TOOLS.md"}


@dataclass
class BootstrapFile:
    """A loaded bootstrap file with metadata."""

    name: str
    path: Path
    content: str
    original_size: int = 0
    truncated: bool = False


@dataclass
class BootstrapContext:
    """All loaded bootstrap files ready for prompt injection."""

    files: list[BootstrapFile] = field(default_factory=list)
    total_chars: int = 0
    has_soul: bool = False

    def get_file(self, name: str) -> BootstrapFile | None:
        for f in self.files:
            if f.name.lower() == name.lower():
                return f
        return None


def _truncate_content(content: str, max_chars: int, head_ratio: float, tail_ratio: float) -> str:
    """Truncate content preserving head and tail portions."""
    if len(content) <= max_chars:
        return content

    head_size = int(max_chars * head_ratio)
    tail_size = int(max_chars * tail_ratio)
    middle_msg = f"\n\n... [{len(content)} chars, truncated — read the full file for complete content] ...\n\n"

    return content[:head_size] + middle_msg + content[-tail_size:]


def load_bootstrap_files(
    workspace_dir: Path,
    config: BootstrapConfig,
    is_subagent: bool = False,
    context_mode: str = "full",  # "full" | "lightweight"
) -> BootstrapContext:
    """Load bootstrap files from workspace directory.

    Args:
        workspace_dir: The agent workspace directory.
        config: Bootstrap configuration (char limits, ratios).
        is_subagent: If True, only load AGENTS.md and TOOLS.md.
        context_mode: "full" loads all files, "lightweight" only HEARTBEAT.md.
    """
    ctx = BootstrapContext()
    total_chars = 0

    # Determine which files to look for
    if context_mode == "lightweight":
        allowed = {"HEARTBEAT.md"}
    elif is_subagent:
        allowed = SUBAGENT_ALLOWED
    else:
        allowed = set(BOOTSTRAP_FILENAMES)

    for filename in BOOTSTRAP_FILENAMES:
        if filename not in allowed:
            continue

        file_path = workspace_dir / filename
        if not file_path.is_file():
            continue

        try:
            raw_content = file_path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue

        if not raw_content.strip():
            continue  # skip blank files

        original_size = len(raw_content)

        # Per-file truncation
        content = _truncate_content(
            raw_content,
            config.max_chars_per_file,
            config.head_ratio,
            config.tail_ratio,
        )

        # Total budget check
        if total_chars + len(content) > config.max_chars_total:
            remaining = config.max_chars_total - total_chars
            if remaining <= 0:
                break
            content = _truncate_content(
                content, remaining, config.head_ratio, config.tail_ratio
            )

        bf = BootstrapFile(
            name=filename,
            path=file_path,
            content=content,
            original_size=original_size,
            truncated=len(content) < original_size,
        )
        ctx.files.append(bf)
        total_chars += len(content)

        if filename.lower() == "soul.md":
            ctx.has_soul = True

    ctx.total_chars = total_chars
    return ctx
