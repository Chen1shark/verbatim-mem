"""classify_intent 与 _temporal_alpha(intent_temporal=)。"""

from app.intent import (
    CHOICE,
    CURRENT_STATE,
    DIRECT_FACT,
    HISTORICAL_STATE,
    LIST_OR_COUNT,
    PREFERENCE,
    classify_intent,
)
from app.store import TIME_WEIGHT, TIME_WEIGHT_TEMPORAL, _temporal_alpha


def test_classify_current_live_without_now() -> None:
    """Where does X live → current_state。"""
    assert classify_intent("Where does the torque wrench live?") == CURRENT_STATE
    assert classify_intent("Where do I live?") == CURRENT_STATE


def test_classify_current_now() -> None:
    """now → current_state。"""
    assert classify_intent("Where do I live now?") == CURRENT_STATE


def test_classify_historical_before_and_grow_up() -> None:
    """before / grow up → historical_state。"""
    assert classify_intent("Where did I live before?") == HISTORICAL_STATE
    assert classify_intent("Where did I grow up?") == HISTORICAL_STATE


def test_classify_direct_fact() -> None:
    """无状态谓词 → direct_fact。"""
    assert classify_intent("When was the autoclave calibrated?") == DIRECT_FACT
    assert classify_intent("What is the spare gasket SKU?") == DIRECT_FACT


def test_classify_preference_and_choice() -> None:
    """like → preference；options 非空 → choice。"""
    assert classify_intent("What music do I like?") == PREFERENCE
    assert (
        classify_intent("Which equipment ratings are recorded?", ["A. 15000 rpm"])
        == CHOICE
    )


def test_classify_list_or_count() -> None:
    """how many → list_or_count。"""
    assert classify_intent("How many rotors are rated?") == LIST_OR_COUNT


def test_temporal_alpha_present_live_is_positive() -> None:
    """intent_temporal=True 时 Where does X live → TIME_WEIGHT_TEMPORAL。"""
    assert (
        _temporal_alpha("Where does the torque wrench live?")
        == TIME_WEIGHT_TEMPORAL
    )


def test_temporal_alpha_intent_off_ignores_live() -> None:
    """intent_temporal=False 时无 _TEMPORAL_* 词 → TIME_WEIGHT。"""
    assert (
        _temporal_alpha(
            "Where does the torque wrench live?", intent_temporal=False
        )
        == TIME_WEIGHT
    )
    assert TIME_WEIGHT == 0.0
