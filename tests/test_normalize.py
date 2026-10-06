from __future__ import annotations

import hashlib

from systemone.normalize.pipeline import bucket
from systemone.normalize.sources import _mcq, _slug, glaive, sciq
from systemone.normalize.text import clean, provenance
from systemone.records import parse


def test_clean_undoes_source_artifacts() -> None:
    assert clean("Wall Street#39;s dwindling\\band") == "Wall Street#39;s dwindling\\band"
    assert clean("Wall Street#39;s dwindling\\band", escapes=True, markup=True) == "Wall Street's dwindling band"
    assert clean('Great!\\n\\nSaid \\"wow\\"', escapes=True) == 'Great!\n\nSaid "wow"'
    assert clean("Fine.<br /><br />Really &amp; truly.", markup=True) == "Fine.\n\nReally & truly."
    assert clean("  spaced \t out \n\n\n\n end\x07 ") == "spaced out\n\nend"
    assert clean("A quot;campquot; camp; day", markup=True) == 'A "campquot; camp; day'


def test_provenance_matches_kev() -> None:
    expected = hashlib.sha256(b"hello world").hexdigest()
    assert provenance({"text": "  Hello   WORLD "}) == expected
    assert provenance({"premise": "Hello world", "hypothesis": "x"}) == expected


def test_mcq_collapses_repeated_distractors_and_refuses_repeated_answers() -> None:
    record = _mcq({"question": "q"}, ["a", "b", "b", "c"], 3)
    assert record is not None
    question = parse(record, "r").questions[0]
    assert question.options == ("opt_1: a", "opt_2: b", "opt_3: c") and question.label == 2
    assert _mcq({"question": "q"}, ["a", "b", "b"], 1) is None
    assert _mcq({"question": "q"}, ["a", "b"], None) is None


def test_slug() -> None:
    assert _slug("MeanOfTransportation") == "mean_of_transportation"
    assert _slug("Criminal Planning/Confessions") == "criminal_planning_confessions"


def test_sciq_answer_is_not_always_first() -> None:
    positions = set()
    for i in range(20):
        row = {
            "question": f"q{i}",
            "correct_answer": "right",
            "distractor1": "a",
            "distractor2": "b",
            "distractor3": "c",
        }
        record = sciq(row, {})
        assert record is not None
        positions.add(parse(record, "r").questions[0].label)
    assert len(positions) > 1


def _glaive_row(called: str | None) -> dict[str, object]:
    system = (
        "SYSTEM: You are a helpful assistant with access to the following functions. Use them if required -\n"
        '{"name": "get_weather", "description": "Weather now", "parameters": {"properties": {"city": {}}}}\n\n'
        '{"name": "get_time", "description": "Time now", "parameters": {"properties": {}}}'
    )
    reply = f'<functioncall> {{"name": "{called}", "arguments": "{{}}"}}' if called else "I can't do that."
    return {"system": system, "chat": f"USER: What's it like in Oslo?\n\n\nASSISTANT: {reply}\n\n\nUSER: thanks"}


def test_glaive_decisions() -> None:
    called = glaive(_glaive_row("get_weather"), {})
    assert called is not None
    call, function = parse(called, "r").questions
    assert call.label == 1 and function.keys[function.label] == "get_weather"
    declined = parse(glaive(_glaive_row(None), {}), "r").questions
    assert declined[0].label == 0 and declined[1].keys[declined[1].label] == "none"
    assert glaive(_glaive_row("unknown_function"), {}) is None


def test_bucket_is_stable_and_uniform() -> None:
    values = [bucket(str(i)) for i in range(2000)]
    assert bucket("x") == bucket("x") and all(0 <= v < 1 for v in values)
    assert 0.45 < sum(v < 0.5 for v in values) / len(values) < 0.55
