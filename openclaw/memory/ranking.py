"""Scoring, MMR diversity re-ranking, and temporal decay for search results."""

from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass

import numpy as np

from openclaw.memory.store import MemoryChunk


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
    """Convert BM25 rank to 0..1 score.

    FTS5 BM25 returns negative ranks where more negative = more relevant.
    Matches OpenClaw: negative rank → relevance/(1+relevance).
    """
    if not math.isfinite(rank):
        return 1.0 / (1.0 + 999.0)
    if rank < 0:
        relevance = -rank
        return relevance / (1.0 + relevance)
    return 1.0 / (1.0 + rank)


# ---------------------------------------------------------------------------
# Jaccard similarity
# ---------------------------------------------------------------------------

def _tokenize_for_jaccard(text: str) -> set[str]:
    """Tokenize text into alphanumeric+underscore tokens for Jaccard similarity."""
    return set(re.findall(r"[a-z0-9_]+", text.lower()))


def _jaccard_similarity(a: str, b: str) -> float:
    """Jaccard text similarity on tokenized content."""
    tokens_a = _tokenize_for_jaccard(a)
    tokens_b = _tokenize_for_jaccard(b)
    if not tokens_a and not tokens_b:
        return 1.0
    if not tokens_a or not tokens_b:
        return 0.0
    smaller, larger = (tokens_a, tokens_b) if len(tokens_a) <= len(tokens_b) else (tokens_b, tokens_a)
    intersection_size = sum(1 for t in smaller if t in larger)
    union_size = len(tokens_a) + len(tokens_b) - intersection_size
    return intersection_size / union_size if union_size > 0 else 0.0


# ---------------------------------------------------------------------------
# Score normalization
# ---------------------------------------------------------------------------

def _normalize_scores(results: list[SearchResult]) -> None:
    """Min-max normalize final_score to [0,1] in-place."""
    if len(results) <= 1:
        return
    min_score = min(r.final_score for r in results)
    max_score = max(r.final_score for r in results)
    score_range = max_score - min_score
    if score_range == 0:
        for r in results:
            r.final_score = 1.0
        return
    for r in results:
        r.final_score = (r.final_score - min_score) / score_range


# ---------------------------------------------------------------------------
# MMR (Maximal Marginal Relevance) re-ranking
# ---------------------------------------------------------------------------

def apply_mmr(
    results: list[SearchResult],
    lambda_param: float = 0.7,
    max_results: int = 10,
) -> list[SearchResult]:
    """Maximal Marginal Relevance re-ranking for diversity.

    λ × relevance − (1−λ) × max_similarity_to_selected

    Uses original (pre-normalization) score as tiebreaker.
    """
    if len(results) <= 1:
        return results

    clamped_lambda = max(0.0, min(1.0, lambda_param))

    if clamped_lambda == 1.0:
        return sorted(results, key=lambda r: r.final_score, reverse=True)[:max_results]

    # Save original scores for tiebreaking
    original_scores: dict[int, float] = {id(r): r.final_score for r in results}

    _normalize_scores(results)

    # Pre-tokenize for efficiency
    token_cache: dict[int, set[str]] = {id(r): _tokenize_for_jaccard(r.chunk.text) for r in results}

    selected: list[SearchResult] = []
    remaining = list(results)

    while remaining and len(selected) < max_results:
        best_idx = -1
        best_mmr = -float("inf")
        best_original_score = -float("inf")

        for i, candidate in enumerate(remaining):
            relevance = candidate.final_score

            max_sim = 0.0
            candidate_tokens = token_cache[id(candidate)]
            for sel in selected:
                sel_tokens = token_cache[id(sel)]
                if not candidate_tokens and not sel_tokens:
                    sim = 1.0
                elif not candidate_tokens or not sel_tokens:
                    sim = 0.0
                else:
                    smaller, larger = (candidate_tokens, sel_tokens) if len(candidate_tokens) <= len(sel_tokens) else (sel_tokens, candidate_tokens)
                    isect = sum(1 for t in smaller if t in larger)
                    union = len(candidate_tokens) + len(sel_tokens) - isect
                    sim = isect / union if union > 0 else 0.0
                max_sim = max(max_sim, sim)

            mmr = clamped_lambda * relevance - (1 - clamped_lambda) * max_sim
            orig_score = original_scores[id(candidate)]

            if mmr > best_mmr or (mmr == best_mmr and orig_score > best_original_score):
                best_mmr = mmr
                best_idx = i
                best_original_score = orig_score

        if best_idx >= 0:
            selected.append(remaining.pop(best_idx))
        else:
            break

    # Restore original scores
    for r in selected:
        r.final_score = original_scores[id(r)]

    return selected


# ---------------------------------------------------------------------------
# Temporal decay
# ---------------------------------------------------------------------------

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
        fp = result.chunk.file_path
        if evergreen_paths and fp in evergreen_paths:
            continue
        if "MEMORY.md" in fp:
            continue

        age_days = (now - result.chunk.updated_at) / 86400.0
        if age_days > 0:
            decay = math.exp(-decay_lambda * age_days)
            result.final_score *= decay

    return results


# ---------------------------------------------------------------------------
# Result clamping
# ---------------------------------------------------------------------------

def clamp_results_by_chars(
    results: list[SearchResult],
    char_budget: int = 8000,
) -> list[SearchResult]:
    """Limit results so total injected text stays within *char_budget*."""
    clamped: list[SearchResult] = []
    total_chars = 0
    for r in results:
        text_len = len(r.chunk.text)
        if total_chars + text_len > char_budget and clamped:
            break
        clamped.append(r)
        total_chars += text_len
    return clamped
