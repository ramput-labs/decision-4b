from __future__ import annotations

import pytest

from systemone.records import Json, RecordError, parse, render

STATE = "I was charged twice for order 1182."


def _record(question: dict[str, Json]) -> Json:
    return {"state": STATE, "questions": {"q": question}}


def test_render_matches_kev() -> None:
    assert render({"ticket": {"channel": "email", "body": "hi"}}) == "ticket:\n  channel: email\n  body: hi"
    assert render([{"role": "customer", "content": "x"}]) == "- role: customer\n  content: x"
    assert render(None) == ""


def test_choice_label_is_option_index() -> None:
    record = parse(
        _record(
            {
                "type": "choice",
                "instructions": "Team?",
                "criteria": {"billing": "Refunds", "shipping": None},
                "label": "shipping",
            }
        ),
        "r",
    )
    q = record.questions[0]
    assert q.options == ("billing: Refunds", "shipping") and q.label == 1


def test_noul_and_score() -> None:
    noul = parse(_record({"type": "noul", "instructions": "Urgent?", "label": True}), "r").questions[0]
    assert noul.options == ("no", "yes") and noul.label == 1
    score = parse(_record({"type": "score", "instructions": "Stars?", "criteria": ["low", "high"], "label": 0}), "r")
    assert score.questions[0].keys == ("0", "1")


def test_soft_target_is_normalized() -> None:
    q = parse(
        _record(
            {
                "type": "choice",
                "instructions": "?",
                "criteria": {"a": None, "b": None},
                "label": "a",
                "target": {"a": 1, "b": 1},
            }
        ),
        "r",
    ).questions[0]
    assert q.target == (0.5, 0.5)


@pytest.mark.parametrize(
    "question",
    [
        {"type": "choice", "instructions": "?", "criteria": {"a": None}, "label": "b"},
        {"type": "score", "instructions": "?", "criteria": ["x"], "label": 1},
        {"type": "score", "instructions": "?", "criteria": ["x", "y"], "label": True},
        {"type": "noul", "instructions": "?", "label": 1},
        {"type": "noul", "instructions": "?", "criteria": {"maybe": "x"}, "label": True},
        {"type": "choice", "instructions": "", "criteria": {"a": None}, "label": "a"},
        {"type": "choice", "instructions": "?", "criteria": {"a": None}, "label": "a", "target": {"z": 1}},
        {"type": "rank", "instructions": "?", "label": 0},
    ],
)
def test_invalid_questions_are_refused(question: dict[str, Json]) -> None:
    with pytest.raises(RecordError):
        parse(_record(question), "r")


def test_empty_state_is_refused() -> None:
    with pytest.raises(RecordError):
        parse({"state": " ", "questions": {"q": {"type": "noul", "instructions": "?", "label": True}}}, "r")
