"""classify_intent：current_state / historical_state / preference / list_or_count / choice / direct_fact。"""

from __future__ import annotations

import re

CURRENT_STATE = "current_state"
HISTORICAL_STATE = "historical_state"
PREFERENCE = "preference"
LIST_OR_COUNT = "list_or_count"
CHOICE = "choice"
DIRECT_FACT = "direct_fact"

_FTS_TOKEN = re.compile(r"[A-Za-z0-9]+")
_TEMPORAL_NEW_EN = frozenset(
    {
        "now",
        "currently",
        "already",
        "latest",
        "recently",
        "anymore",
        "nowadays",
        "lately",
        "still",
    }
)
_TEMPORAL_OLD_EN = frozenset(
    {
        "previous",
        "previously",
        "before",
        "earlier",
        "formerly",
    }
)
_TEMPORAL_NEW_PHRASES = (
    "right now",
    "these days",
    "as of now",
)
_TEMPORAL_OLD_PHRASES = (
    "used to",
    "no longer",
    "back then",
    "at the time",
)
_GROW_UP_PHRASES = (
    "grow up",
    "grew up",
    "grown up",
    "growing up",
    "childhood",
)
_PREF_QUERY_EN = frozenset(
    {
        "like",
        "likes",
        "love",
        "loves",
        "prefer",
        "prefers",
        "favorite",
        "favourite",
        "enjoy",
        "enjoys",
        "hate",
        "hates",
        "dislike",
        "dislikes",
    }
)
_WHERE_PRESENT = re.compile(r"\bwhere\s+(do|does|is|are|am)\b", re.IGNORECASE)
_PAST_AUX = re.compile(r"\b(did|was|were|had)\b", re.IGNORECASE)
_STATE_VERB = re.compile(
    r"\b(live|lives|living|lived|stay|stays|staying|stayed|work|works|working|worked|located)\b",
    re.IGNORECASE,
)
_PRESENT_STATE_VERB = re.compile(
    r"\b(live|lives|living|stay|stays|staying|work|works|working|located)\b",
    re.IGNORECASE,
)
_LIST_PHRASE = re.compile(r"\b(how many|how much|list all|list the)\b", re.IGNORECASE)


def _tokens(query: str) -> set[str]:
    """query 的 [A-Za-z0-9]+ 小写集合。"""
    return {token.lower() for token in _FTS_TOKEN.findall(query)}


def _has_new(query: str, tokens: set[str]) -> bool:
    """tokens ∩ _TEMPORAL_NEW_EN 或 _TEMPORAL_NEW_PHRASES。"""
    lowered = query.lower()
    return bool(tokens & _TEMPORAL_NEW_EN) or any(
        phrase in lowered for phrase in _TEMPORAL_NEW_PHRASES
    )


def _has_old(query: str, tokens: set[str]) -> bool:
    """tokens ∩ _TEMPORAL_OLD_EN 或 _TEMPORAL_OLD_PHRASES。"""
    lowered = query.lower()
    return bool(tokens & _TEMPORAL_OLD_EN) or any(
        phrase in lowered for phrase in _TEMPORAL_OLD_PHRASES
    )


def classify_intent(query: str, options: list[str] | None = None) -> str:
    """规则路由：grow-up/旧时态 → historical_state；now/现在时状态谓词 → current_state。"""
    lowered = query.lower()
    tokens = _tokens(query)
    old = _has_old(query, tokens)
    new = _has_new(query, tokens)
    if any(phrase in lowered for phrase in _GROW_UP_PHRASES):
        return HISTORICAL_STATE
    if old and not new:
        return HISTORICAL_STATE
    if new:
        return CURRENT_STATE
    if _PAST_AUX.search(lowered) and _STATE_VERB.search(lowered):
        return HISTORICAL_STATE
    if _WHERE_PRESENT.search(lowered) or _PRESENT_STATE_VERB.search(lowered):
        return CURRENT_STATE
    if tokens & _PREF_QUERY_EN:
        return PREFERENCE
    if _LIST_PHRASE.search(lowered) or "count" in tokens:
        return LIST_OR_COUNT
    if options:
        return CHOICE
    return DIRECT_FACT
