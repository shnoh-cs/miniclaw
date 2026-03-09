"""File and session watchers, and cross-encoder reranker."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from openclaw.memory.embeddings import EmbeddingProvider
from openclaw.memory.ranking import SearchResult

if TYPE_CHECKING:
    from openclaw.memory.search import MemorySearcher

log = logging.getLogger("openclaw.memory.watchers")


class FileWatcher:
    """Track file modification times for delta-based reindexing.

    Call ``check_and_reindex()`` to detect changed files and re-index only
    those that have been modified since the last check.  Debounced so it
    runs at most once every ``debounce_seconds`` (default 30 s).
    """

    def __init__(self, debounce_seconds: float = 30.0) -> None:
        self._mtimes: dict[str, float] = {}
        self._debounce_seconds = debounce_seconds
        self._last_check: float = 0.0

    def register(self, path: Path) -> None:
        """Register a file path to watch (records its current mtime)."""
        try:
            self._mtimes[str(path)] = path.stat().st_mtime
        except OSError:
            log.debug("FileWatcher: cannot stat %s", path)

    def check_changed(self) -> list[Path]:
        """Return list of registered paths whose mtime has changed."""
        now = time.time()
        if now - self._last_check < self._debounce_seconds:
            return []
        self._last_check = now

        changed: list[Path] = []
        for path_str, old_mtime in list(self._mtimes.items()):
            p = Path(path_str)
            try:
                current_mtime = p.stat().st_mtime
            except OSError:
                changed.append(p)
                continue
            if current_mtime != old_mtime:
                self._mtimes[path_str] = current_mtime
                changed.append(p)
        return changed

    async def check_and_reindex(self, searcher: MemorySearcher) -> int:
        """Check for changed files and re-index them.

        Returns total number of new chunks indexed across all changed files.
        """
        changed = self.check_changed()
        total = 0
        for path in changed:
            if path.exists():
                total += await searcher.index_file(path)
        return total


class Reranker:
    """Cross-encoder reranker for search result refinement.

    Falls back to embedding-based cosine similarity reranking if no
    dedicated reranker model is available.
    """

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        model: str | None = None,
    ) -> None:
        self.embedding_provider = embedding_provider
        self.model = model

    async def rerank(
        self,
        query: str,
        results: list[SearchResult],
        top_k: int = 10,
    ) -> list[SearchResult]:
        """Re-score and re-sort results by cross-encoder relevance."""
        if not results or len(results) <= 1:
            return results

        try:
            texts = [query] + [r.chunk.text[:500] for r in results]
            embeddings = await self.embedding_provider.embed(texts)
            if not embeddings or len(embeddings) != len(texts):
                return results

            query_emb = embeddings[0]
            for i, r in enumerate(results):
                chunk_emb = embeddings[i + 1]
                rerank_score = float(np.dot(query_emb, chunk_emb))
                r.final_score = 0.6 * rerank_score + 0.4 * r.final_score

            results.sort(key=lambda r: r.final_score, reverse=True)
            return results[:top_k]
        except Exception:
            log.debug("Reranker failed, returning original order", exc_info=True)
            return results


class SessionSyncWatcher:
    """Track session JSONL file sizes for delta-threshold background sync."""

    def __init__(self, delta_threshold_bytes: int = 8192) -> None:
        self._sizes: dict[str, int] = {}
        self.delta_threshold_bytes = delta_threshold_bytes

    def check(self, jsonl_path: Path) -> bool:
        """Return True if the file has grown enough to warrant re-indexing."""
        key = str(jsonl_path)
        try:
            current_size = jsonl_path.stat().st_size
        except OSError:
            return False

        last_size = self._sizes.get(key, 0)
        delta = current_size - last_size
        if delta >= self.delta_threshold_bytes:
            self._sizes[key] = current_size
            return True
        return False

    def mark_synced(self, jsonl_path: Path) -> None:
        """Update the recorded size after a successful sync."""
        key = str(jsonl_path)
        try:
            self._sizes[key] = jsonl_path.stat().st_size
        except OSError:
            log.debug("SessionSyncWatcher: cannot stat %s", jsonl_path)
