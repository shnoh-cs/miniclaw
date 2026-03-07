"""Built-in tool: apply structured patches across files."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from openclaw.agent.types import ToolDefinition, ToolParameter, ToolResult

DEFINITION = ToolDefinition(
    name="apply_patch",
    description="Apply a unified diff patch to one or more files. Supports standard unified diff format.",
    parameters=[
        ToolParameter(name="patch", description="The unified diff patch content"),
        ToolParameter(name="cwd", description="Working directory for relative paths", required=False),
    ],
)


async def execute(args: dict[str, Any], workspace: str = "") -> ToolResult:
    patch_text = args.get("patch", "")
    cwd = args.get("cwd", workspace) or workspace

    if not patch_text:
        return ToolResult(tool_use_id="", content="Error: patch is required", is_error=True)

    # Parse unified diff
    files_patched = 0
    errors: list[str] = []

    # Split into per-file chunks
    file_diffs = re.split(r"(?=^--- )", patch_text, flags=re.MULTILINE)

    for diff in file_diffs:
        diff = diff.strip()
        if not diff:
            continue

        # Extract file paths
        old_match = re.search(r"^--- (?:a/)?(.+)$", diff, re.MULTILINE)
        new_match = re.search(r"^\+\+\+ (?:b/)?(.+)$", diff, re.MULTILINE)
        if not old_match or not new_match:
            continue

        target_path = new_match.group(1).strip()
        if target_path == "/dev/null":
            target_path = old_match.group(1).strip()

        full_path = Path(cwd) / target_path if cwd else Path(target_path)

        # Extract hunks
        hunks = re.findall(
            r"^@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@.*?\n((?:[+ \-].*\n?)*)",
            diff,
            re.MULTILINE,
        )

        if not hunks:
            errors.append(f"No hunks found for {target_path}")
            continue

        try:
            if full_path.exists():
                lines = full_path.read_text(encoding="utf-8").splitlines(keepends=True)
            else:
                lines = []
                full_path.parent.mkdir(parents=True, exist_ok=True)

            # Apply hunks in reverse order to preserve line numbers
            for old_start, new_start, hunk_body in reversed(hunks):
                old_idx = int(old_start) - 1
                hunk_lines = hunk_body.splitlines(keepends=True)

                # Calculate removals and additions
                remove_count = sum(1 for l in hunk_lines if l.startswith("-"))
                new_lines = [l[1:] for l in hunk_lines if l.startswith("+")]
                context_and_remove = [l for l in hunk_lines if l.startswith("-") or l.startswith(" ")]

                # Remove old lines and insert new
                del lines[old_idx:old_idx + len(context_and_remove)]
                for j, new_line in enumerate(new_lines):
                    lines.insert(old_idx + j, new_line)

            full_path.write_text("".join(lines), encoding="utf-8")
            files_patched += 1

        except Exception as e:
            errors.append(f"Error patching {target_path}: {e}")

    if errors:
        error_text = "\n".join(errors)
        if files_patched > 0:
            return ToolResult(
                tool_use_id="",
                content=f"Patched {files_patched} file(s) with errors:\n{error_text}",
            )
        return ToolResult(tool_use_id="", content=f"Patch failed:\n{error_text}", is_error=True)

    return ToolResult(tool_use_id="", content=f"Successfully patched {files_patched} file(s)")
