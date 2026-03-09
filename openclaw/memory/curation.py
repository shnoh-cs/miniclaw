"""Memory curation: promote recurring patterns from daily notes to MEMORY.md.

Two-phase approach:
1. Embedding-based: find semantically similar paragraphs across different days
   (cross-day repetition), filter against MEMORY.md for novelty
2. Model-based: synthesize the novel recurring themes into concise bullet points

Falls back to prompt-only if embeddings are unavailable.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from openclaw.memory.embeddings import EmbeddingProvider
    from openclaw.model.provider import ModelProvider

logger = logging.getLogger(__name__)

# Minimum interval between curation runs (24 hours)
CURATION_INTERVAL_SECONDS = 86400

_CURATION_MARKER = ".last-curation"

# Cosine similarity thresholds
_CROSS_DAY_SIM_THRESHOLD = 0.75  # min similarity to consider "recurring"
_NOVELTY_SIM_THRESHOLD = 0.80  # max similarity to MEMORY.md to be "novel"


def _should_curate(workspace_dir: str) -> bool:
    """Check if enough time has passed since last curation."""
    marker = Path(workspace_dir) / _CURATION_MARKER
    if not marker.exists():
        return True
    try:
        last_time = float(marker.read_text().strip())
        return (time.time() - last_time) >= CURATION_INTERVAL_SECONDS
    except (ValueError, OSError):
        return True


def _mark_curated(workspace_dir: str) -> None:
    """Update the curation timestamp marker."""
    marker = Path(workspace_dir) / _CURATION_MARKER
    try:
        marker.write_text(str(time.time()))
    except OSError:
        logger.debug("Failed to write curation marker", exc_info=True)


async def _find_recurring_themes(
    embedding_provider: EmbeddingProvider,
    daily_notes: list[Path],
) -> list[str]:
    """Find semantically recurring themes across daily notes using embeddings.

    Embeds paragraphs from each daily note, computes cross-day cosine
    similarity, and returns paragraphs that appear (semantically) in
    multiple days.
    """
    if len(daily_notes) < 2:
        return []

    # Read and chunk each daily note into paragraphs
    all_texts: list[str] = []
    all_dates: list[str] = []

    for note in daily_notes:
        try:
            text = note.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        paragraphs = [p.strip() for p in text.split("\n\n") if len(p.strip()) > 50]
        for para in paragraphs[:10]:
            all_texts.append(para[:500])
            all_dates.append(note.stem)

    if len(all_texts) < 4:
        return []

    # Batch embed
    embeddings = await embedding_provider.embed(all_texts)
    if not embeddings or len(embeddings) != len(all_texts):
        return []

    emb_matrix = np.vstack(embeddings)  # already L2-normalized by provider

    # Compute pairwise cosine similarity (dot product for unit vectors)
    sim_matrix = emb_matrix @ emb_matrix.T

    # Find cross-day high-similarity pairs
    recurring: set[int] = set()
    n = len(all_texts)
    for i in range(n):
        for j in range(i + 1, n):
            if all_dates[i] != all_dates[j] and sim_matrix[i][j] > _CROSS_DAY_SIM_THRESHOLD:
                recurring.add(i)
                recurring.add(j)

    return [all_texts[i] for i in sorted(recurring)][:20]


async def _filter_novel(
    embedding_provider: EmbeddingProvider,
    candidates: list[str],
    workspace_dir: str,
) -> list[str]:
    """Filter out candidates already covered in MEMORY.md using embeddings."""
    memory_md = Path(workspace_dir) / "MEMORY.md"
    if not memory_md.exists():
        return candidates  # all novel

    try:
        existing = memory_md.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return candidates

    existing_lines = [
        line.strip() for line in existing.split("\n") if len(line.strip()) > 20
    ]
    if not existing_lines:
        return candidates

    # Embed candidates and existing lines together
    all_texts = candidates + existing_lines
    embeddings = await embedding_provider.embed(all_texts)
    if not embeddings or len(embeddings) != len(all_texts):
        return candidates

    emb_matrix = np.vstack(embeddings)
    n_cand = len(candidates)
    cand_embs = emb_matrix[:n_cand]
    exist_embs = emb_matrix[n_cand:]

    # Cosine similarity: candidates vs existing
    sim = cand_embs @ exist_embs.T
    max_sim = sim.max(axis=1) if sim.shape[1] > 0 else np.zeros(n_cand)

    # Keep only candidates with low similarity to existing (novel)
    return [c for c, s in zip(candidates, max_sim) if s < _NOVELTY_SIM_THRESHOLD]


async def curate_memories(
    memory_dir: Path,
    workspace_dir: str,
    provider: ModelProvider,
    model: str,
    *,
    embedding_provider: EmbeddingProvider | None = None,
    max_daily_notes: int = 7,
) -> bool:
    """Scan daily notes for recurring patterns and promote to MEMORY.md.

    Two-phase approach when embedding_provider is available:
    1. Embedding: find cross-day recurring themes, filter for novelty
    2. Model: synthesize novel themes into MEMORY.md entries

    Falls back to prompt-only when embeddings are unavailable.

    Returns True if any content was promoted.
    """
    if not _should_curate(workspace_dir):
        return False

    from openclaw.agent.types import AgentMessage, TextBlock

    # Find recent daily notes
    daily_notes = sorted(
        [f for f in memory_dir.glob("????-??-??.md") if f.is_file()],
        key=lambda f: f.stem,
        reverse=True,
    )[:max_daily_notes]

    if not daily_notes:
        _mark_curated(workspace_dir)
        return False

    # Read current MEMORY.md
    memory_md = Path(workspace_dir) / "MEMORY.md"
    existing = ""
    if memory_md.exists():
        try:
            existing = memory_md.read_text(encoding="utf-8", errors="replace")
        except OSError:
            logger.debug("Failed to read MEMORY.md", exc_info=True)

    # Phase 1: Embedding-based candidate detection (if available)
    embedding_candidates: list[str] | None = None
    if embedding_provider:
        try:
            recurring = await _find_recurring_themes(embedding_provider, daily_notes)
            if recurring:
                novel = await _filter_novel(
                    embedding_provider, recurring, workspace_dir
                )
                if novel:
                    embedding_candidates = novel
                    logger.info(
                        "Curation: %d recurring themes found, %d novel",
                        len(recurring), len(novel),
                    )
        except Exception as exc:
            logger.warning("Embedding-based curation failed, falling back: %s", exc)

    # Phase 2: Build prompt
    if embedding_candidates:
        # Embedding-guided: model synthesizes from pre-filtered candidates
        candidates_text = "\n\n".join(
            f"- {c}" for c in embedding_candidates[:15]
        )
        prompt = (
            "The following themes were detected as recurring across multiple "
            "daily notes and are NOT yet in MEMORY.md.\n\n"
            f"Recurring themes:\n{candidates_text}\n\n"
            f"Current MEMORY.md:\n```\n{existing[:1500]}\n```\n\n"
            "Synthesize these into concise bullet points for long-term memory.\n"
            "Output ONLY the new lines to append. If nothing worth promoting, "
            "reply with exactly: NO_REPLY"
        )
    else:
        # Fallback: prompt-only (read all daily notes)
        daily_content = ""
        for note in daily_notes:
            try:
                text = note.read_text(encoding="utf-8", errors="replace")
                daily_content += f"\n## {note.stem}\n{text[:1500]}\n"
            except OSError:
                continue

        if not daily_content.strip():
            _mark_curated(workspace_dir)
            return False

        prompt = (
            "Review these daily memory notes and the current long-term memory.\n\n"
            f"Current MEMORY.md:\n```\n{existing[:2000]}\n```\n\n"
            f"Recent daily notes:\n{daily_content[:4000]}\n\n"
            "Identify recurring patterns, user preferences, or important decisions "
            "that appear across multiple days and should be promoted.\n"
            "Output ONLY the new bullet points to append to MEMORY.md.\n"
            "Do NOT duplicate content already in MEMORY.md.\n"
            "If nothing qualifies for promotion, reply with exactly: NO_REPLY"
        )

    try:
        response = await provider.complete(
            system=(
                "You are a memory curator. Extract durable patterns from daily "
                "notes into concise bullet points for long-term memory."
            ),
            messages=[AgentMessage(role="user", content=[TextBlock(text=prompt)])],
        )
    except Exception as exc:
        logger.warning("Memory curation failed: %s", exc)
        _mark_curated(workspace_dir)
        return False

    text = response.strip()
    if text.upper() == "NO_REPLY" or not text:
        _mark_curated(workspace_dir)
        return False

    # Append to MEMORY.md
    try:
        with open(memory_md, "a", encoding="utf-8") as f:
            f.write(f"\n\n## Auto-Promoted ({daily_notes[0].stem})\n")
            f.write(text)
            f.write("\n")
        logger.info("Memory curation: promoted %d chars to MEMORY.md", len(text))
    except OSError as exc:
        logger.warning("Memory curation write failed: %s", exc)
        return False

    _mark_curated(workspace_dir)
    return True
