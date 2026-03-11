"""Query expansion and FTS5 query building with multi-language stop words."""

from __future__ import annotations

import re

# --------------------------------------------------------------------------- #
# Multi-language stop words (mirrors OpenClaw query-expansion.ts)
# --------------------------------------------------------------------------- #

_STOP_WORDS_EN: set[str] = {
    "a", "an", "the", "this", "that", "these", "those",
    "i", "me", "my", "we", "our", "you", "your", "he", "she", "it", "they", "them",
    "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did",
    "will", "would", "could", "should", "can", "may", "might",
    "in", "on", "at", "to", "for", "of", "with", "by", "from",
    "about", "into", "through", "during", "before", "after",
    "above", "below", "between", "under", "over",
    "and", "or", "but", "if", "then", "because", "as", "while",
    "when", "where", "what", "which", "who", "how", "why",
    "yesterday", "today", "tomorrow", "earlier", "later",
    "recently", "ago", "just", "now",
    "thing", "things", "stuff", "something", "anything", "everything", "nothing",
    "please", "help", "find", "show", "get", "tell", "give",
}

_STOP_WORDS_KO: set[str] = {
    "은", "는", "이", "가", "을", "를", "의", "에", "에서",
    "로", "으로", "와", "과", "도", "만", "까지", "부터",
    "한테", "에게", "께", "처럼", "같이", "보다", "마다", "밖에", "대로",
    "나", "나는", "내가", "나를", "너", "우리", "저", "저희",
    "그", "그녀", "그들", "이것", "저것", "그것", "여기", "저기", "거기",
    "있다", "없다", "하다", "되다", "이다", "아니다",
    "보다", "주다", "오다", "가다",
    "것", "거", "등", "수", "때", "곳", "중", "분",
    "잘", "더", "또", "매우", "정말", "아주", "많이", "너무", "좀",
    "그리고", "하지만", "그래서", "그런데", "그러나", "또는", "그러면",
    "왜", "어떻게", "뭐", "언제", "어디", "누구", "무엇", "어떤",
    "어제", "오늘", "내일", "최근", "지금", "아까", "나중", "전에",
    "제발", "부탁",
}

_STOP_WORDS_ZH: set[str] = {
    "我", "我们", "你", "你们", "他", "她", "它", "他们",
    "这", "那", "这个", "那个", "这些", "那些",
    "的", "了", "着", "过", "得", "地", "吗", "呢", "吧", "啊", "呀", "嘛", "啦",
    "是", "有", "在", "被", "把", "给", "让", "用", "到", "去", "来", "做", "说",
    "看", "找", "想", "要", "能", "会", "可以",
    "和", "与", "或", "但", "但是", "因为", "所以", "如果", "虽然",
    "而", "也", "都", "就", "还", "又", "再", "才", "只",
    "之前", "以前", "之后", "以后", "刚才", "现在", "昨天", "今天", "明天", "最近",
    "东西", "事情", "事", "什么", "哪个", "哪些", "怎么", "为什么", "多少",
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

_ALL_STOP_WORDS: set[str] = _STOP_WORDS_EN | _STOP_WORDS_KO | _STOP_WORDS_ZH | _STOP_WORDS_JA

# Korean trailing particles sorted by descending length for longest-match-first
_KO_TRAILING_PARTICLES: list[str] = sorted(
    [
        "에서", "으로", "에게", "한테", "처럼", "같이", "보다", "까지", "부터",
        "마다", "밖에", "대로",
        "은", "는", "이", "가", "을", "를", "의", "에", "로", "와", "과", "도", "만",
    ],
    key=len,
    reverse=True,
)

# Regex for CJK character ranges
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
    if re.match(r"^[a-zA-Z]+$", token) and len(token) < 3:
        return False
    if re.match(r"^\d+$", token):
        return False
    if re.match(r"^[\W]+$", token) and not _CJK_RANGE.search(token):
        return False
    return True


def _tokenize_query(text: str) -> list[str]:
    """Tokenize text for query expansion, handling CJK scripts."""
    tokens: list[str] = []
    normalized = text.lower().strip()
    segments = re.split(r"[\s\.,;:!?\-\(\)\[\]{}<>\"'`~@#$%^&*_+=|/\\]+", normalized)

    for segment in segments:
        if not segment:
            continue

        if _HANGUL_RANGE.search(segment):
            stem = _strip_korean_trailing_particle(segment)
            stem_is_stop = stem is not None and stem in _STOP_WORDS_KO
            if segment not in _STOP_WORDS_KO and not stem_is_stop:
                tokens.append(segment)
            if stem and stem not in _STOP_WORDS_KO and _is_useful_korean_stem(stem):
                tokens.append(stem)
        elif _CJK_UNIFIED.search(segment):
            chars = [c for c in segment if _CJK_UNIFIED.match(c)]
            tokens.extend(chars)
            for i in range(len(chars) - 1):
                tokens.append(chars[i] + chars[i + 1])
        else:
            tokens.append(segment)

    return tokens


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


def build_fts_query(raw: str) -> str | None:
    """Build a safe FTS5 query from raw user input.

    Tokenizes with a proper pattern (alphanumeric + underscore sequences),
    wraps each token in quotes, and joins with AND.
    Returns None if no valid tokens are found.
    """
    tokens = re.findall(r"[\w\u4e00-\u9fff\uac00-\ud7af\u3040-\u30ff]+", raw, re.UNICODE)
    tokens = [t.strip() for t in tokens if t.strip()]
    if not tokens:
        return None
    quoted = ['"' + t.replace('"', '""') + '"' for t in tokens]
    return " AND ".join(quoted)
