from __future__ import annotations

import hashlib
import json
from pathlib import Path

from den.api import Json, parse
from den.normalize import bucket
from den.sources import GLAIVE_RAW, _mcq, _slug, _words, glaive, sciq
from den.text import clean, provenance


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


SYSTEM = (
    "SYSTEM: You are a helpful assistant with access to the following functions. Use them if required -\n"
    '{"name": "get_weather", "description": "Weather now", "parameters": {"properties": {"city": {}}}}\n\n'
    '{"name": "get_time", "description": "Time now", "parameters": {"properties": {}}}'
)
UNRELATED = (
    '{"name": "calculate_tip", "description": "Tip for a bill", "parameters": {"properties": {"bill": {}}}}\n\n'
    '{"name": "play_music", "description": "Play a song", "parameters": {"properties": {"song": {}}}}'
)


def _call(name: str) -> str:
    return f'<functioncall> {{"name": "{name}", "arguments": "{{}}"}}'


def _glaive_row(*replies: str, request: str = "What's the weather like?") -> dict[str, object]:
    users = [request, "In Oslo.", "Thanks", "Bye"]
    chat = "\n\n\n".join(f"USER: {u}\n\n\nASSISTANT: {r}" for u, r in zip(users, replies, strict=False))
    return {"system": SYSTEM, "chat": chat}


def _glaive_root(tmp_path: Path) -> Path:
    raw = tmp_path / GLAIVE_RAW
    raw.parent.mkdir(parents=True)
    raw.write_text(json.dumps([{"system": SYSTEM}, {"system": UNRELATED}]))
    return tmp_path


def _labels(record: Json) -> dict[str, str | int]:
    assert record is not None
    return {q.id: q.keys[q.label] if q.type == "choice" else q.label for q in parse(record, "r").questions}


def _kept(root: Path, *replies: str, request: str = "What's the weather like?") -> list[dict[str, str | int]]:
    """Labels of the same conversation over many requests, swapped or not."""
    convert = glaive(root)
    return [_labels(convert(_glaive_row(*replies, request=f"{request} #{i}"), {})) for i in range(40)]


def test_glaive_decisions(tmp_path: Path) -> None:
    root = _glaive_root(tmp_path)
    called = _kept(root, _call("get_weather"))
    assert {"call": 1, "ready": 1, "function": "get_weather"} in called
    assert all(
        c in ({"call": 1, "ready": 1, "function": "get_weather"}, {"call": 0, "function": "none"}) for c in called
    )
    assert all(c == {"call": 0, "function": "none"} for c in _kept(root, "I can't do that.", "Sorry."))
    assert glaive(root)(_glaive_row(_call("unknown_function")), {}) is None


def test_glaive_asking_for_arguments_is_not_declining(tmp_path: Path) -> None:
    root = _glaive_root(tmp_path)
    asked = _kept(root, "Which city?", _call("get_weather"), "You're welcome.")
    assert {"call": 1, "ready": 0, "function": "get_weather"} in asked
    assert glaive(root)(_glaive_row("Sorry, I can't. I can tell the time.", _call("get_time")), {}) is None
    assert glaive(root)(_glaive_row("Which city?", "And which day?", _call("get_weather")), {}) is None


def test_glaive_swaps_some_answered_requests_to_unrelated_functions(tmp_path: Path) -> None:
    convert, called = glaive(_glaive_root(tmp_path)), {"name": "get_weather", "description": "Weather now"}
    swapped = []
    for i in range(200):
        record = convert(_glaive_row(_call("get_weather"), request=f"Weather please #{i}"), {})
        assert isinstance(record, dict)
        state = record["state"]
        assert isinstance(state, dict)
        if _labels(record)["call"] == 0:
            functions = state["functions"]
            assert isinstance(functions, list)
            words = _words(called) | _words({"description": f"Weather please #{i}"})
            assert not any(words & _words(f) for f in functions if isinstance(f, dict))
            swapped.append(record)
    assert 0.15 < len(swapped) / 200 < 0.35
    assert convert(_glaive_row(_call("get_weather"), request="Weather please #7"), {}) == convert(
        _glaive_row(_call("get_weather"), request="Weather please #7"), {}
    )


def test_bucket_is_stable_and_uniform() -> None:
    values = [bucket(str(i)) for i in range(2000)]
    assert bucket("x") == bucket("x") and all(0 <= v < 1 for v in values)
    assert 0.45 < sum(v < 0.5 for v in values) / len(values) < 0.55


def test_clean_drops_zero_width_but_keeps_emoji_joiners() -> None:
    assert clean("French​ pronunciation﻿") == "French pronunciation"
    assert clean("\U0001f937‍♀️") == "\U0001f937‍♀️"
    assert clean("Tom &amp; Jerry &lt;br&gt; end", markup=True) == "Tom & Jerry\nend"
