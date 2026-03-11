"""Hybrid search: BM25 + vector with MMR diversity and temporal decay.

Core MemorySearcher class. Scoring, query expansion, and watchers are split
into ``ranking``, ``query``, and ``watchers`` submodules.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from openclaw.config import MemoryConfig
from openclaw.memory.embeddings import EmbeddingProvider
from openclaw.memory.store import MemoryChunk, MemoryStore

# Re-export public API so existing ``from openclaw.memory.search import X``
# continues to work after the split.
from openclaw.memory.ranking import (  # noqa: F401
    SearchResult,
    apply_mmr,
    apply_temporal_decay,
    bm25_rank_to_score,
    clamp_results_by_chars,
    cosine_similarity,
    _jaccard_similarity,
    _normalize_scores,
    _tokenize_for_jaccard,
)
from openclaw.memory.query import (  # noqa: F401
    build_fts_query,
    expand_query,
    _tokenize_query,
    _ALL_STOP_WORDS,
)
from openclaw.memory.watchers import (  # noqa: F401
    FileWatcher,
    Reranker,
    SessionSyncWatcher,
)

log = logging.getLogger("openclaw.memory.search")

# Candidate multiplier: fetch max_results * N candidates for hybrid search
_CANDIDATE_MULTIPLIER = 4


class MemorySearcher:
    """Hybrid search engine combining vector similarity and BM25."""

    def __init__(
        self,
        store: MemoryStore,
        embedding_provider: EmbeddingProvider,
        config: MemoryConfig,
        file_watcher: FileWatcher | None = None,
        reranker: Reranker | None = None,
    ) -> None:
        self.store = store
        self.embedding_provider = embedding_provider
        self.config = config
        self.file_watcher = file_watcher
        self.reranker = reranker
        self.session_sync = SessionSyncWatcher()

    async def search(
        self,
        query: str,
        max_results: int = 10,
    ) -> list[SearchResult]:
        """Perform hybrid search (vector + BM25) with MMR and temporal decay."""
        if not query or not query.strip():
            return []

        # Auto-reindex changed files (debounced)
        if self.file_watcher is not None:
            try:
                await self.file_watcher.check_and_reindex(self)
            except Exception:
                log.debug("FileWatcher reindex failed", exc_info=True)

        hybrid = self.config.hybrid
        candidates = min(200, max(1, max_results * _CANDIDATE_MULTIPLIER))

        results_by_id: dict[int, SearchResult] = {}

        # 1. Vector search
        try:
            query_embedding = await self.embedding_provider.embed_single(query)
            cached = self.store.get_all_embeddings_cached()

            if cached is not None:
                ids, matrix = cached
                query_norm = np.linalg.norm(query_embedding)
                if query_norm > 0:
                    row_norms = np.linalg.norm(matrix, axis=1)
                    safe_norms = np.where(row_norms > 0, row_norms, 1.0)
                    similarities = matrix @ query_embedding / (safe_norms * query_norm)
                    if len(similarities) > candidates:
                        top_indices = np.argpartition(similarities, -candidates)[-candidates:]
                        top_indices = top_indices[np.argsort(similarities[top_indices])[::-1]]
                    else:
                        top_indices = np.argsort(similarities)[::-1]

                    chunk_ids = [ids[i] for i in top_indices]
                    chunks = self.store.get_chunks_by_ids(chunk_ids)
                    chunk_map = {c.id: c for c in chunks}

                    for idx in top_indices:
                        cid = ids[idx]
                        sim = float(similarities[idx])
                        if cid in chunk_map:
                            results_by_id[cid] = SearchResult(
                                chunk=chunk_map[cid],
                                vector_score=sim,
                            )
        except Exception:
            log.debug("Vector search failed, falling back to BM25 only", exc_info=True)

        # 2. BM25 keyword search
        if hybrid.enabled:
            fts_query = build_fts_query(query)
            if fts_query is None:
                expanded = expand_query(query)
                fts_query = build_fts_query(expanded)

            if fts_query:
                bm25_results = self.store.bm25_search(fts_query, limit=candidates)
            else:
                bm25_results = []

            if bm25_results:
                bm25_ids = [cid for cid, _ in bm25_results]
                chunks = self.store.get_chunks_by_ids(bm25_ids)
                chunk_map = {c.id: c for c in chunks}

                for chunk_id, rank in bm25_results:
                    score = bm25_rank_to_score(rank)
                    if chunk_id in results_by_id:
                        results_by_id[chunk_id].bm25_score = score
                    elif chunk_id in chunk_map:
                        results_by_id[chunk_id] = SearchResult(
                            chunk=chunk_map[chunk_id],
                            bm25_score=score,
                        )

        # 3. Merge scores
        vw = hybrid.vector_weight
        tw = hybrid.text_weight
        for result in results_by_id.values():
            result.final_score = vw * result.vector_score + tw * result.bm25_score

        all_results = sorted(results_by_id.values(), key=lambda r: r.final_score, reverse=True)

        # 4. MMR re-ranking
        if hybrid.mmr.enabled:
            all_results = apply_mmr(
                all_results,
                lambda_param=hybrid.mmr.lambda_param,
                max_results=max_results,
            )

        # 5. Temporal decay
        if hybrid.temporal_decay.enabled:
            all_results = apply_temporal_decay(
                all_results,
                half_life_days=hybrid.temporal_decay.half_life_days,
            )
            all_results.sort(key=lambda r: r.final_score, reverse=True)

        # 6. Reranker pass
        if self.reranker and all_results:
            all_results = await self.reranker.rerank(
                query, all_results, top_k=max_results
            )

        return all_results[:max_results]

    async def index_file(self, file_path: Path, chunk_size: int = 1600, overlap: int = 320) -> int:
        """Index a markdown file into the store. Returns number of chunks created."""
        if not file_path.exists():
            return 0

        text = file_path.read_text(encoding="utf-8", errors="replace")
        if not text.strip():
            return 0

        self.store.delete_by_file(str(file_path))

        lines = text.splitlines()
        chunks: list[tuple[int, int, str]] = []

        i = 0
        while i < len(lines):
            end = min(i + chunk_size // 50, len(lines))
            chunk_text = "\n".join(lines[i:end])

            if len(chunk_text) > chunk_size:
                chunk_text = chunk_text[:chunk_size]
                last_nl = chunk_text.rfind("\n")
                if last_nl > chunk_size // 2:
                    chunk_text = chunk_text[:last_nl]
                    end = i + chunk_text.count("\n") + 1

            if chunk_text.strip():
                chunks.append((i + 1, end, chunk_text))

            overlap_lines = max(1, overlap // 50)
            next_i = end - overlap_lines if end < len(lines) else len(lines)
            i = max(i + 1, next_i)  # always advance at least 1 line

        if not chunks:
            return 0

        texts = [c[2] for c in chunks]
        embeddings = await self._batch_embed_with_cache(texts)

        for (line_start, line_end, chunk_text), emb in zip(chunks, embeddings):
            chunk = MemoryChunk(
                file_path=str(file_path),
                line_start=line_start,
                line_end=line_end,
                text=chunk_text,
                embedding=emb,
            )
            self.store.upsert_chunk(chunk)

        return len(chunks)

    async def sync_session_if_needed(self, jsonl_path: Path) -> int:
        """Re-index a session JSONL if it has grown past the delta threshold."""
        if not self.session_sync.check(jsonl_path):
            return 0
        count = await self.index_session_jsonl(jsonl_path)
        self.session_sync.mark_synced(jsonl_path)
        return count

    async def index_session_jsonl(
        self,
        jsonl_path: Path,
        chunk_size: int = 1600,
        overlap: int = 320,
    ) -> int:
        """Index a JSONL session file into memory with source_type='session'."""
        if not jsonl_path.exists():
            return 0

        texts: list[str] = []
        try:
            with open(jsonl_path, encoding="utf-8", errors="replace") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    role = entry.get("role", "")
                    if role not in ("user", "assistant"):
                        continue
                    content = entry.get("content", "")
                    if isinstance(content, str) and content.strip():
                        texts.append(content.strip())
                    elif isinstance(content, list):
                        for block in content:
                            if isinstance(block, dict) and block.get("type") == "text":
                                t = block.get("text", "")
                                if t.strip():
                                    texts.append(t.strip())
        except OSError:
            log.warning("Cannot read session JSONL: %s", jsonl_path)
            return 0

        if not texts:
            return 0

        self.store.delete_by_file(str(jsonl_path))

        merged = "\n\n".join(texts)
        lines = merged.splitlines()
        raw_chunks: list[tuple[int, int, str]] = []

        i = 0
        while i < len(lines):
            end = min(i + chunk_size // 50, len(lines))
            chunk_text = "\n".join(lines[i:end])

            if len(chunk_text) > chunk_size:
                chunk_text = chunk_text[:chunk_size]
                last_nl = chunk_text.rfind("\n")
                if last_nl > chunk_size // 2:
                    chunk_text = chunk_text[:last_nl]
                    end = i + chunk_text.count("\n") + 1

            if chunk_text.strip():
                raw_chunks.append((i + 1, end, chunk_text))

            overlap_lines = max(1, overlap // 50)
            next_i = end - overlap_lines if end < len(lines) else len(lines)
            i = max(i + 1, next_i)  # always advance at least 1 line

        if not raw_chunks:
            return 0

        chunk_texts = [c[2] for c in raw_chunks]
        embeddings = await self._batch_embed_with_cache(chunk_texts)

        for (line_start, line_end, chunk_text), emb in zip(raw_chunks, embeddings):
            chunk = MemoryChunk(
                file_path=str(jsonl_path),
                line_start=line_start,
                line_end=line_end,
                text=chunk_text,
                embedding=emb,
                source_type="session",
            )
            self.store.upsert_chunk(chunk)

        return len(raw_chunks)

    # -------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------

    async def _batch_embed_with_cache(
        self, texts: list[str]
    ) -> list[np.ndarray | None]:
        """Embed texts using cache first, then batch-embed uncached ones."""
        embeddings: list[np.ndarray | None] = [None] * len(texts)
        uncached_indices: list[int] = []
        uncached_texts: list[str] = []

        for idx, t in enumerate(texts):
            text_hash = EmbeddingProvider.text_hash(t)
            cached = self.store.get_cached_embedding(text_hash)
            if cached is not None:
                embeddings[idx] = cached
            else:
                uncached_indices.append(idx)
                uncached_texts.append(t)

        if uncached_texts:
            new_embeddings = await self.embedding_provider.embed(uncached_texts)
            for i, emb in zip(uncached_indices, new_embeddings):
                embeddings[i] = emb
                text_hash = EmbeddingProvider.text_hash(texts[i])
                self.store.cache_embedding(text_hash, emb)

        return embeddings
