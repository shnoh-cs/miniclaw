"""Hybrid search: BM25 + vector with MMR diversity and temporal decay."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from openclaw.config import MemoryConfig
from openclaw.memory.embeddings import EmbeddingProvider
from openclaw.memory.store import MemoryChunk, MemoryStore


@dataclass
class SearchResult:
    """A single search result with scores."""

    chunk: MemoryChunk
    vector_score: float = 0.0
    bm25_score: float = 0.0
    final_score: float = 0.0


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def bm25_rank_to_score(rank: float) -> float:
    """Convert BM25 rank to 0..1 score."""
    return 1.0 / (1.0 + max(0, abs(rank)))


def _jaccard_similarity(a: str, b: str) -> float:
    """Jaccard text similarity on tokenized content."""
    tokens_a = set(a.lower().split())
    tokens_b = set(b.lower().split())
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = tokens_a & tokens_b
    union = tokens_a | tokens_b
    return len(intersection) / len(union)


def apply_mmr(
    results: list[SearchResult],
    lambda_param: float = 0.7,
    max_results: int = 10,
) -> list[SearchResult]:
    """Maximal Marginal Relevance re-ranking for diversity.

    λ × relevance − (1−λ) × max_similarity_to_selected
    """
    if len(results) <= 1:
        return results

    selected: list[SearchResult] = []
    remaining = list(results)

    while remaining and len(selected) < max_results:
        best_idx = -1
        best_mmr = -float("inf")

        for i, candidate in enumerate(remaining):
            relevance = candidate.final_score

            # Max similarity to already selected
            max_sim = 0.0
            for sel in selected:
                sim = _jaccard_similarity(candidate.chunk.text, sel.chunk.text)
                max_sim = max(max_sim, sim)

            mmr = lambda_param * relevance - (1 - lambda_param) * max_sim
            if mmr > best_mmr:
                best_mmr = mmr
                best_idx = i

        if best_idx >= 0:
            selected.append(remaining.pop(best_idx))
        else:
            break

    return selected


def apply_temporal_decay(
    results: list[SearchResult],
    half_life_days: int = 30,
    evergreen_paths: set[str] | None = None,
) -> list[SearchResult]:
    """Apply temporal decay to scores based on chunk age.

    score × e^(−λ × ageInDays)
    Evergreen files (MEMORY.md, non-dated) never decay.
    """
    if not results:
        return results

    decay_lambda = math.log(2) / half_life_days
    now = time.time()

    for result in results:
        # Check if evergreen
        fp = result.chunk.file_path
        if evergreen_paths and fp in evergreen_paths:
            continue
        if "MEMORY.md" in fp:
            continue
        # Check if it's a dated file (memory/YYYY-MM-DD.md)
        # Non-dated memory files are evergreen

        age_days = (now - result.chunk.updated_at) / 86400.0
        if age_days > 0:
            decay = math.exp(-decay_lambda * age_days)
            result.final_score *= decay

    return results


def expand_query(query: str) -> str:
    """Simple keyword expansion for improved search.

    Extracts potential keywords from the query for BM25 matching.
    """
    # Remove common stop words and extract meaningful terms
    stop_words = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been",
        "being", "have", "has", "had", "do", "does", "did", "will",
        "would", "could", "should", "may", "might", "can", "shall",
        "to", "of", "in", "for", "on", "with", "at", "by", "from",
        "as", "into", "about", "like", "through", "after", "before",
        "between", "under", "above", "up", "down", "out", "off",
        "over", "then", "than", "that", "this", "these", "those",
        "what", "which", "who", "when", "where", "how", "why",
        "and", "or", "but", "not", "no", "if", "it", "its", "my",
        "your", "his", "her", "our", "their",
    }

    words = query.lower().split()
    keywords = [w for w in words if w not in stop_words and len(w) > 2]
    return " ".join(keywords) if keywords else query


class MemorySearcher:
    """Hybrid search engine combining vector similarity and BM25."""

    def __init__(
        self,
        store: MemoryStore,
        embedding_provider: EmbeddingProvider,
        config: MemoryConfig,
    ) -> None:
        self.store = store
        self.embedding_provider = embedding_provider
        self.config = config

    async def search(
        self,
        query: str,
        max_results: int = 10,
    ) -> list[SearchResult]:
        """Perform hybrid search (vector + BM25) with MMR and temporal decay."""
        hybrid = self.config.hybrid

        results_by_id: dict[int, SearchResult] = {}

        # 1. Vector search
        try:
            query_embedding = await self.embedding_provider.embed_single(query)
            all_embeddings = self.store.get_all_embeddings()

            if all_embeddings:
                # Compute cosine similarities
                scores = []
                for chunk_id, emb in all_embeddings:
                    sim = cosine_similarity(query_embedding, emb)
                    scores.append((chunk_id, sim))

                scores.sort(key=lambda x: x[1], reverse=True)
                top_vector = scores[:max_results * 2]

                chunk_ids = [cid for cid, _ in top_vector]
                chunks = self.store.get_chunks_by_ids(chunk_ids)
                chunk_map = {c.id: c for c in chunks}

                for chunk_id, sim in top_vector:
                    if chunk_id in chunk_map:
                        results_by_id[chunk_id] = SearchResult(
                            chunk=chunk_map[chunk_id],
                            vector_score=sim,
                        )
        except Exception:
            pass  # degrade to BM25 only

        # 2. BM25 keyword search
        if hybrid.enabled:
            expanded_query = expand_query(query)
            bm25_results = self.store.bm25_search(expanded_query, limit=max_results * 2)

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

        # Sort by final score
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

        return all_results[:max_results]

    async def index_file(self, file_path: Path, chunk_size: int = 700, overlap: int = 80) -> int:
        """Index a markdown file into the store. Returns number of chunks created."""
        if not file_path.exists():
            return 0

        text = file_path.read_text(encoding="utf-8", errors="replace")
        if not text.strip():
            return 0

        # Delete existing chunks for this file
        self.store.delete_by_file(str(file_path))

        # Split into chunks
        lines = text.splitlines()
        chunks: list[tuple[int, int, str]] = []  # (line_start, line_end, text)

        i = 0
        while i < len(lines):
            end = min(i + chunk_size // 50, len(lines))  # rough line-based chunking
            chunk_text = "\n".join(lines[i:end])

            if len(chunk_text) > chunk_size:
                # Split by character limit
                chunk_text = chunk_text[:chunk_size]
                # Find last newline within limit
                last_nl = chunk_text.rfind("\n")
                if last_nl > chunk_size // 2:
                    chunk_text = chunk_text[:last_nl]
                    end = i + chunk_text.count("\n") + 1

            if chunk_text.strip():
                chunks.append((i + 1, end, chunk_text))

            # Advance with overlap
            overlap_lines = max(1, overlap // 50)
            i = end - overlap_lines if end < len(lines) else len(lines)

        if not chunks:
            return 0

        # Batch embed
        texts = [c[2] for c in chunks]

        # Check cache first
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

        # Embed uncached texts
        if uncached_texts:
            new_embeddings = await self.embedding_provider.embed(uncached_texts)
            for i, emb in zip(uncached_indices, new_embeddings):
                embeddings[i] = emb
                # Cache the embedding
                text_hash = EmbeddingProvider.text_hash(texts[i])
                self.store.cache_embedding(text_hash, emb)

        # Store chunks
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
