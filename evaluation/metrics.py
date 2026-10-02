"""Deterministic retrieval and answer-quality metrics (no LLM required)."""

import re
from collections.abc import Sequence

from app.prompts import NO_ANSWER

_WORD_RE = re.compile(r"[a-z0-9%]+")
_CITATION_RE = re.compile(r"\[(S?\d+)\]")
_STOP = frozenset(
    "a an and are as at be by for from has have in is it of on or that the this to was with "
    "you your can should will not do if".split()
)


def hit_at_k(retrieved: Sequence[str], relevant: Sequence[str], k: int) -> float:
    if not relevant:
        return float("nan")
    return 1.0 if set(retrieved[:k]) & set(relevant) else 0.0


def recall_at_k(retrieved: Sequence[str], relevant: Sequence[str], k: int) -> float:
    if not relevant:
        return float("nan")
    return len(set(retrieved[:k]) & set(relevant)) / len(set(relevant))


def precision_at_k(retrieved: Sequence[str], relevant: Sequence[str], k: int) -> float:
    top = list(retrieved[:k])
    if not relevant or not top:
        return float("nan")
    return sum(1 for r in top if r in set(relevant)) / len(top)


def reciprocal_rank(retrieved: Sequence[str], relevant: Sequence[str]) -> float:
    if not relevant:
        return float("nan")
    for rank, doc in enumerate(retrieved, start=1):
        if doc in set(relevant):
            return 1.0 / rank
    return 0.0


def key_fact_coverage(answer: str, key_facts: Sequence[str]) -> float:
    """Fraction of expected key facts mentioned in the answer (case-insensitive substring)."""
    if not key_facts:
        return float("nan")
    text = answer.lower()
    return sum(1 for f in key_facts if f.lower() in text) / len(key_facts)


def is_abstention(answer: str) -> bool:
    return NO_ANSWER.lower().rstrip(".") in answer.lower()


def citation_validity(answer: str, valid_labels: Sequence[str]) -> float:
    """Fraction of citations in the answer that point at a passage actually provided."""
    cited = _CITATION_RE.findall(answer)
    if not cited:
        return float("nan")
    valid = set(valid_labels)
    return sum(1 for c in cited if c in valid) / len(cited)


def _content_words(text: str) -> set[str]:
    return {w for w in _WORD_RE.findall(text.lower()) if w not in _STOP and len(w) > 2}


def unsupported_sentence_ratio(
    answer: str, contexts: Sequence[str], threshold: float = 0.5
) -> float:
    """Hallucination proxy: share of answer sentences whose content words are mostly absent
    from the retrieved context. 0.0 = fully grounded, 1.0 = nothing grounded."""
    context_words = set().union(*(_content_words(c) for c in contexts)) if contexts else set()
    sentences = [s for s in re.split(r"(?<=[.!?])\s+|\n+", answer) if _content_words(s)]
    sentences = [s for s in sentences if not is_abstention(s) and "LLM unavailable" not in s]
    if not sentences:
        return 0.0
    unsupported = 0
    for s in sentences:
        words = _content_words(_CITATION_RE.sub("", s))
        if words and len(words & context_words) / len(words) < threshold:
            unsupported += 1
    return unsupported / len(sentences)


def mean(values: Sequence[float]) -> float:
    vals = [v for v in values if v == v]  # drop NaN
    return round(sum(vals) / len(vals), 4) if vals else float("nan")
