"""Hybrid search: BM25 + vector with MMR diversity and temporal decay."""

from __future__ import annotations

import json
import math
import re
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from openclaw.config import MemoryConfig
from openclaw.memory.embeddings import EmbeddingProvider
from openclaw.memory.store import MemoryChunk, MemoryStore

# Candidate multiplier: fetch max_results * N candidates for hybrid search
# before MMR re-ranking. Matches the original OpenClaw default of 4.
_CANDIDATE_MULTIPLIER = 4


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
    normalized = max(0, abs(rank)) if math.isfinite(rank) else 999.0
    return 1.0 / (1.0 + normalized)


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
    # Iterate over the smaller set for efficiency
    smaller, larger = (tokens_a, tokens_b) if len(tokens_a) <= len(tokens_b) else (tokens_b, tokens_a)
    intersection_size = sum(1 for t in smaller if t in larger)
    union_size = len(tokens_a) + len(tokens_b) - intersection_size
    return intersection_size / union_size if union_size > 0 else 0.0


def _normalize_scores(results: list[SearchResult]) -> None:
    """Min-max normalize final_score to [0,1] in-place.

    This ensures the relevance vs diversity tradeoff in MMR is fair,
    since Jaccard similarity is already in [0,1].
    """
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


def apply_mmr(
    results: list[SearchResult],
    lambda_param: float = 0.7,
    max_results: int = 10,
) -> list[SearchResult]:
    """Maximal Marginal Relevance re-ranking for diversity.

    Scores are min-max normalized to [0,1] before applying MMR so that the
    relevance vs diversity tradeoff is balanced (Jaccard is already [0,1]).

    λ × relevance − (1−λ) × max_similarity_to_selected

    Uses original (pre-normalization) score as tiebreaker when MMR scores
    are equal, matching the original OpenClaw implementation.
    """
    if len(results) <= 1:
        return results

    # Clamp lambda to valid range
    clamped_lambda = max(0.0, min(1.0, lambda_param))

    # If lambda is 1, just return sorted by relevance (no diversity penalty)
    if clamped_lambda == 1.0:
        return sorted(results, key=lambda r: r.final_score, reverse=True)[:max_results]

    # Save original scores for tiebreaking before normalization
    original_scores: dict[int, float] = {}
    for r in results:
        original_scores[id(r)] = r.final_score

    # Min-max normalize to [0,1] for fair comparison with Jaccard similarity
    _normalize_scores(results)

    # Pre-tokenize all items for efficiency
    token_cache: dict[int, set[str]] = {}
    for r in results:
        token_cache[id(r)] = _tokenize_for_jaccard(r.chunk.text)

    selected: list[SearchResult] = []
    remaining = list(results)

    while remaining and len(selected) < max_results:
        best_idx = -1
        best_mmr = -float("inf")
        best_original_score = -float("inf")

        for i, candidate in enumerate(remaining):
            relevance = candidate.final_score

            # Max similarity to already selected
            max_sim = 0.0
            candidate_tokens = token_cache[id(candidate)]
            for sel in selected:
                sel_tokens = token_cache[id(sel)]
                # Inline Jaccard for cached tokens
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

            # Use original score as tiebreaker when MMR scores are equal
            if mmr > best_mmr or (mmr == best_mmr and orig_score > best_original_score):
                best_mmr = mmr
                best_idx = i
                best_original_score = orig_score

        if best_idx >= 0:
            selected.append(remaining.pop(best_idx))
        else:
            break

    # Restore original scores so callers see meaningful values
    for r in selected:
        r.final_score = original_scores[id(r)]

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


# --------------------------------------------------------------------------- #
# Multi-language stop words (mirrors OpenClaw query-expansion.ts)
# --------------------------------------------------------------------------- #

_STOP_WORDS_EN: set[str] = {
    # Articles and determiners
    "a", "an", "the", "this", "that", "these", "those",
    # Pronouns
    "i", "me", "my", "we", "our", "you", "your", "he", "she", "it", "they", "them",
    # Common verbs
    "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did",
    "will", "would", "could", "should", "can", "may", "might",
    # Prepositions
    "in", "on", "at", "to", "for", "of", "with", "by", "from",
    "about", "into", "through", "during", "before", "after",
    "above", "below", "between", "under", "over",
    # Conjunctions
    "and", "or", "but", "if", "then", "because", "as", "while",
    "when", "where", "what", "which", "who", "how", "why",
    # Time references (vague)
    "yesterday", "today", "tomorrow", "earlier", "later",
    "recently", "ago", "just", "now",
    # Vague references
    "thing", "things", "stuff", "something", "anything", "everything", "nothing",
    # Question/request words
    "please", "help", "find", "show", "get", "tell", "give",
}

_STOP_WORDS_KO: set[str] = {
    # Particles (조사)
    "은", "는", "이", "가", "을", "를", "의", "에", "에서",
    "로", "으로", "와", "과", "도", "만", "까지", "부터",
    "한테", "에게", "께", "처럼", "같이", "보다", "마다", "밖에", "대로",
    # Pronouns (대명사)
    "나", "나는", "내가", "나를", "너", "우리", "저", "저희",
    "그", "그녀", "그들", "이것", "저것", "그것", "여기", "저기", "거기",
    # Common verbs / auxiliaries
    "있다", "없다", "하다", "되다", "이다", "아니다",
    "보다", "주다", "오다", "가다",
    # Nouns (의존 명사 / vague)
    "것", "거", "등", "수", "때", "곳", "중", "분",
    # Adverbs
    "잘", "더", "또", "매우", "정말", "아주", "많이", "너무", "좀",
    # Conjunctions
    "그리고", "하지만", "그래서", "그런데", "그러나", "또는", "그러면",
    # Question words
    "왜", "어떻게", "뭐", "언제", "어디", "누구", "무엇", "어떤",
    # Time (vague)
    "어제", "오늘", "내일", "최근", "지금", "아까", "나중", "전에",
    # Request words
    "제발", "부탁",
}

_STOP_WORDS_ZH: set[str] = {
    # Pronouns
    "我", "我们", "你", "你们", "他", "她", "它", "他们",
    "这", "那", "这个", "那个", "这些", "那些",
    # Auxiliary words
    "的", "了", "着", "过", "得", "地", "吗", "呢", "吧", "啊", "呀", "嘛", "啦",
    # Common verbs
    "是", "有", "在", "被", "把", "给", "让", "用", "到", "去", "来", "做", "说",
    "看", "找", "想", "要", "能", "会", "可以",
    # Prepositions and conjunctions
    "和", "与", "或", "但", "但是", "因为", "所以", "如果", "虽然",
    "而", "也", "都", "就", "还", "又", "再", "才", "只",
    # Time (vague)
    "之前", "以前", "之后", "以后", "刚才", "现在", "昨天", "今天", "明天", "最近",
    # Vague references
    "东西", "事情", "事", "什么", "哪个", "哪些", "怎么", "为什么", "多少",
    # Request words
    "请", "帮", "帮忙", "告诉",
}

_STOP_WORDS_JA: set[str] = {
    "これ", "それ", "あれ", "この", "その", "あの", "ここ", "そこ", "あそこ",
    "する", "した", "して", "です", "ます", "いる", "ある", "なる", "できる",
    "の", "こと", "もの", "ため", "そして", "しかし", "また", "でも", "から", "まで",
    "より", "だけ",
    "なぜ", "どう", "何", "いつ", "どこ", "誰", "どれ",
    "昨日", "今日", "明日", "最近", "今", "さっき", "前", "後",
}

# All stop words merged for fast lookup
_ALL_STOP_WORDS: set[str] = _STOP_WORDS_EN | _STOP_WORDS_KO | _STOP_WORDS_ZH | _STOP_WORDS_JA

# Korean trailing particles sorted by descending length for longest-match-first stripping
_KO_TRAILING_PARTICLES: list[str] = sorted(
    [
        "에서", "으로", "에게", "한테", "처럼", "같이", "보다", "까지", "부터",
        "마다", "밖에", "대로",
        "은", "는", "이", "가", "을", "를", "의", "에", "로", "와", "과", "도", "만",
    ],
    key=len,
    reverse=True,
)

# Regex for CJK character ranges (unified CJK, Hangul syllables, Hiragana, Katakana)
_CJK_RANGE = re.compile(r"[\u4e00-\u9fff\uac00-\ud7af\u3040-\u30ff]")
_HANGUL_RANGE = re.compile(r"[\uac00-\ud7af\u3131-\u3163]")
_CJK_UNIFIED = re.compile(r"[\u4e00-\u9fff]")


def _strip_korean_trailing_particle(token: str) -> str | None:
    """Strip common Korean particles from word end, returning the stem or None."""
    for particle in _KO_TRAILING_PARTICLES:
        if len(token) > len(particle) and token.endswith(particle):
            return token[: -len(particle)]
    return None


def _is_useful_korean_stem(stem: str) -> bool:
    """Prevent bogus one-syllable stems; keep 2+ syllable Hangul or ASCII."""
    if _HANGUL_RANGE.search(stem):
        return len(stem) >= 2
    return bool(re.match(r"^[a-z0-9_]+$", stem, re.IGNORECASE))


def _is_valid_keyword(token: str) -> bool:
    """Check if a token looks like a meaningful keyword."""
    if not token:
        return False
    # Skip very short English words
    if re.match(r"^[a-zA-Z]+$", token) and len(token) < 3:
        return False
    # Skip pure numbers
    if re.match(r"^\d+$", token):
        return False
    # Skip all-punctuation
    if re.match(r"^[\W]+$", token) and not _CJK_RANGE.search(token):
        return False
    return True


def expand_query(query: str) -> str:
    """Extract keywords from a query for FTS search.

    Handles English, Korean, Chinese, and Japanese text.
    Removes stop words, strips Korean trailing particles,
    and extracts CJK character bigrams for better matching.
    """
    tokens = _tokenize_query(query)
    seen: set[str] = set()
    keywords: list[str] = []

    for token in tokens:
        if token in _ALL_STOP_WORDS:
            continue
        if not _is_valid_keyword(token):
            continue
        if token in seen:
            continue
        seen.add(token)
        keywords.append(token)

    return " ".join(keywords) if keywords else query


def _tokenize_query(text: str) -> list[str]:
    """Tokenize text for query expansion, handling CJK scripts.

    For Chinese characters: extract unigrams + bigrams.
    For Korean: keep words, strip trailing particles, emit stems.
    For other text: split on whitespace/punctuation.
    """
    tokens: list[str] = []
    normalized = text.lower().strip()
    # Split into segments on whitespace and punctuation
    segments = re.split(r"[\s\p{P}]+", normalized, flags=re.UNICODE)

    for segment in segments:
        if not segment:
            continue

        if _HANGUL_RANGE.search(segment):
            # Korean: keep word if not a stop word, also emit particle-stripped stem
            stem = _strip_korean_trailing_particle(segment)
            stem_is_stop = stem is not None and stem in _STOP_WORDS_KO
            if segment not in _STOP_WORDS_KO and not stem_is_stop:
                tokens.append(segment)
            if stem and stem not in _STOP_WORDS_KO and _is_useful_korean_stem(stem):
                tokens.append(stem)
        elif _CJK_UNIFIED.search(segment):
            # Chinese / CJK unified: extract chars + bigrams
            chars = [c for c in segment if _CJK_UNIFIED.match(c)]
            tokens.extend(chars)
            for i in range(len(chars) - 1):
                tokens.append(chars[i] + chars[i + 1])
        else:
            # Non-CJK: keep as single token
            tokens.append(segment)

    return tokens


def build_fts_query(raw: str) -> str | None:
    """Build a safe FTS5 query from raw user input.

    Tokenizes with a proper pattern (alphanumeric + underscore sequences),
    wraps each token in quotes, and joins with AND.
    Returns None if no valid tokens are found.
    """
    # Match alphanumeric + underscore sequences, plus CJK characters
    tokens = re.findall(r"[\w\u4e00-\u9fff\uac00-\ud7af\u3040-\u30ff]+", raw, re.UNICODE)
    tokens = [t.strip() for t in tokens if t.strip()]
    if not tokens:
        return None
    # Quote each token and join with AND for FTS5
    quoted = ['"' + t.replace('"', "") + '"' for t in tokens]
    return " AND ".join(quoted)


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
            pass

    def check_changed(self) -> list[Path]:
        """Return list of registered paths whose mtime has changed.

        Respects the debounce interval — returns an empty list if called
        too soon after the previous check.
        """
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
                # File deleted — mark as changed so caller can handle
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

    Calls an OpenAI-compatible /v1/embeddings endpoint with (query, document)
    pairs to produce relevance scores, similar to how the original OpenClaw
    uses a local GGUF reranker model via QMD.

    Falls back to a lightweight embedding-based reranker if no dedicated
    reranker model is available: computes cosine(query_emb, chunk_emb) and
    uses that as the reranker score.
    """

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        model: str | None = None,
    ) -> None:
        self.embedding_provider = embedding_provider
        self.model = model  # dedicated reranker model (optional)

    async def rerank(
        self,
        query: str,
        results: list[SearchResult],
        top_k: int = 10,
    ) -> list[SearchResult]:
        """Re-score and re-sort results by cross-encoder relevance.

        Uses embedding cosine similarity as a proxy for cross-encoder scoring.
        The query is embedded alongside each chunk text; the cosine similarity
        between query and chunk embeddings serves as the reranker score.
        """
        if not results or len(results) <= 1:
            return results

        try:
            # Embed query + all chunk texts in one batch
            texts = [query] + [r.chunk.text[:500] for r in results]
            embeddings = await self.embedding_provider.embed(texts)
            if not embeddings or len(embeddings) != len(texts):
                return results  # degrade gracefully

            query_emb = embeddings[0]
            for i, r in enumerate(results):
                chunk_emb = embeddings[i + 1]
                rerank_score = float(np.dot(query_emb, chunk_emb))
                # Blend: 60% reranker, 40% original hybrid score
                r.final_score = 0.6 * rerank_score + 0.4 * r.final_score

            results.sort(key=lambda r: r.final_score, reverse=True)
            return results[:top_k]
        except Exception:
            return results  # degrade gracefully


def clamp_results_by_chars(
    results: list[SearchResult],
    char_budget: int = 8000,
) -> list[SearchResult]:
    """Limit results so total injected text stays within *char_budget*.

    Iterates through results (assumed pre-sorted by relevance) and stops
    adding once the cumulative character count would exceed the budget.
    """
    clamped: list[SearchResult] = []
    total_chars = 0
    for r in results:
        text_len = len(r.chunk.text)
        if total_chars + text_len > char_budget and clamped:
            # Already have at least one result; stop here
            break
        clamped.append(r)
        total_chars += text_len
    return clamped


class SessionSyncWatcher:
    """Track session JSONL file sizes for delta-threshold background sync.

    When a session JSONL grows by more than ``delta_threshold_bytes`` since the
    last sync, triggers background re-indexing of that session.
    """

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
            pass


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
        # Guard: return empty results for blank/whitespace queries
        if not query or not query.strip():
            return []

        # Auto-reindex changed files (debounced)
        if self.file_watcher is not None:
            try:
                await self.file_watcher.check_and_reindex(self)
            except Exception:
                pass  # don't let watcher errors block search

        hybrid = self.config.hybrid
        # Use 4x candidate multiplier (matching original OpenClaw)
        candidates = min(200, max(1, max_results * _CANDIDATE_MULTIPLIER))

        results_by_id: dict[int, SearchResult] = {}

        # 1. Vector search (batch cosine similarity via cached matrix)
        try:
            query_embedding = await self.embedding_provider.embed_single(query)
            cached = self.store.get_all_embeddings_cached()

            if cached is not None:
                ids, matrix = cached
                # Batch cosine similarity: dot(query, matrix^T) / (|query| * |rows|)
                query_norm = np.linalg.norm(query_embedding)
                if query_norm > 0:
                    row_norms = np.linalg.norm(matrix, axis=1)
                    # Avoid division by zero for any row
                    safe_norms = np.where(row_norms > 0, row_norms, 1.0)
                    similarities = matrix @ query_embedding / (safe_norms * query_norm)
                    # Get top-k indices efficiently via argpartition
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
            pass  # degrade to BM25 only

        # 2. BM25 keyword search
        if hybrid.enabled:
            # Build a safe FTS5 query; fall back to expanded keywords
            fts_query = build_fts_query(query)
            if fts_query is None:
                # No valid tokens — use expanded keywords as fallback
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

        # 6. Reranker pass (cross-encoder refinement)
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

    async def sync_session_if_needed(self, jsonl_path: Path) -> int:
        """Re-index a session JSONL if it has grown past the delta threshold.

        Called periodically (e.g. per-turn) to keep session memory up to date
        without re-indexing on every single message append.
        Returns number of chunks indexed, or 0 if no sync was needed.
        """
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
        """Index a JSONL session file into memory with source_type='session'.

        Reads each line as JSON, extracts text content from user/assistant
        messages, chunks them, and stores with source_type='session'.
        Returns total number of chunks created.
        """
        if not jsonl_path.exists():
            return 0

        # Extract text from session messages
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
                        # Multi-part content blocks
                        for block in content:
                            if isinstance(block, dict) and block.get("type") == "text":
                                t = block.get("text", "")
                                if t.strip():
                                    texts.append(t.strip())
        except OSError:
            return 0

        if not texts:
            return 0

        # Delete existing session chunks for this file
        self.store.delete_by_file(str(jsonl_path))

        # Merge texts and chunk
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
            i = end - overlap_lines if end < len(lines) else len(lines)

        if not raw_chunks:
            return 0

        # Batch embed
        chunk_texts = [c[2] for c in raw_chunks]
        embeddings: list[np.ndarray | None] = [None] * len(chunk_texts)
        uncached_indices: list[int] = []
        uncached_texts: list[str] = []

        for idx, t in enumerate(chunk_texts):
            text_hash = EmbeddingProvider.text_hash(t)
            cached = self.store.get_cached_embedding(text_hash)
            if cached is not None:
                embeddings[idx] = cached
            else:
                uncached_indices.append(idx)
                uncached_texts.append(t)

        if uncached_texts:
            new_embeddings = await self.embedding_provider.embed(uncached_texts)
            for idx_i, emb in zip(uncached_indices, new_embeddings):
                embeddings[idx_i] = emb
                text_hash = EmbeddingProvider.text_hash(chunk_texts[idx_i])
                self.store.cache_embedding(text_hash, emb)

        # Store chunks with source_type='session'
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
