from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from collections.abc import Iterator
from http.server import ThreadingHTTPServer
from typing import Any

import pytest
import torch

from den.api import Json, Record
from den.runtime import answer, request, respond
from den.serve import handler

KEV_EXAMPLE: dict[str, Json] = {
    "state": "Shoes arrived two weeks late and in the wrong size. Also I see two charges on my card.",
    "model": "kev-latest",
    "questions": {
        "department": {
            "type": "choice",
            "instructions": "Which team should handle this?",
            "criteria": {"returns": "Exchanges", "shipping": "Delivery", "billing": "Charges"},
        },
        "escalate": {"type": "noul", "instructions": "Does this need urgent human attention?"},
        "frustration": {
            "type": "score",
            "instructions": "How frustrated is the customer?",
            "criteria": ["Calm", "Frustrated", "Very angry"],
        },
    },
}


def test_answers_match_kevs_published_example() -> None:
    """Kev's README response: choice confidence 0.21 for (0.47, 0.28, 0.25), score 1.44 and confidence 0.34."""
    department, escalate, frustration = request(KEV_EXAMPLE).questions
    choice = answer(department, [0.47, 0.28, 0.25])
    assert choice["choice"] == "returns" and abs(choice["confidence"] - 0.205) < 1e-3
    assert choice["probabilities"] == {"returns": 0.47, "shipping": 0.28, "billing": 0.25}
    assert answer(escalate, [0.07, 0.93]) == {"type": "noul", "noul": 0.93}
    score = answer(frustration, [0.0, 0.56, 0.44])
    assert score["score"] == 1.44 and abs(score["confidence"] - 0.34) < 1e-3
    assert score["legend"] == {"0": "Calm", "1": "Frustrated", "2": "Very angry"}


class Fake:
    """A stand-in for runtime.Model: favours the second option of every question."""

    temperature = 1.0
    backbone: Any = type("B", (), {"device": type("D", (), {"backend": "cpu", "dtype": "float32"})()})()

    def read(self, record: Record) -> tuple[list[torch.Tensor], int]:
        return [torch.tensor([0.0, 2.0] + [-1.0] * (len(q.options) - 2)) for q in record.questions], 42


def test_respond_has_kevs_response_shape() -> None:
    body = respond(Fake(), KEV_EXAMPLE)  # type: ignore[arg-type]
    assert body["model"] == "kev-latest" and set(body) == {"model", "answers", "usage", "latency_ms"}
    assert body["answers"]["department"]["choice"] == "shipping"
    assert body["answers"]["escalate"]["noul"] > 0.5
    assert body["usage"] == {"input_tokens": 42, "output_tokens": 0}


@pytest.fixture
def server() -> Iterator[str]:
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler(Fake(), "den-test", "secret"))  # type: ignore[arg-type]
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()


def _call(url: str, body: object = None, key: str | None = "secret") -> tuple[int, dict[str, Any]]:
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, headers={"content-type": "application/json"})
    if key:
        req.add_header("authorization", f"Bearer {key}")
    try:
        with urllib.request.urlopen(req) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def test_http_endpoint(server: str) -> None:
    status, body = _call(f"{server}/v1/systemone", KEV_EXAMPLE)
    assert status == 200 and body["answers"]["department"]["choice"] == "shipping"
    assert _call(f"{server}/v1/systemone", KEV_EXAMPLE, key="wrong")[0] == 401
    assert _call(f"{server}/v1/systemone", {"state": "s", "questions": {}})[0] == 400
    assert _call(f"{server}/v1/models")[1]["data"][0]["id"] == "den-test"
    assert _call(f"{server}/health", key=None) == (200, {"status": "ok"})
    assert _call(f"{server}/v1/other", {})[0] == 404


@pytest.mark.parametrize(
    ("body", "message"),
    [
        ({"state": "s", "questions": {"q": {"type": "maybe", "instructions": "?"}}}, "unknown question type"),
        ({"state": "s", "questions": {"q": {"type": "choice", "instructions": "?"}}}, "choice needs"),
        (
            {"state": "s", "questions": {"q": {"type": "choice", "instructions": "", "criteria": {"a": None}}}},
            "empty instructions",
        ),
        (
            {"state": "s", "questions": {"q": {"type": "score", "instructions": "?", "criteria": ["x", "x"]}}},
            "same text",
        ),
        ({"state": "", "questions": {"q": {"type": "noul", "instructions": "?"}}}, "empty state"),
        ({"state": "s", "questions": {}}, "non-empty questions"),
        ({"questions": {"q": {"type": "noul", "instructions": "?"}}}, "no state"),
        ([1, 2], "JSON object"),
    ],
)
def test_bad_requests_get_a_clear_400(server: str, body: object, message: str) -> None:
    status, reply = _call(f"{server}/v1/systemone", body)
    assert status == 400 and message in reply["error"]


def test_any_number_of_options_and_every_type(server: str) -> None:
    questions: dict[str, Json] = {
        f"c{k}": {"type": "choice", "instructions": "Pick", "criteria": {f"o{i}": None for i in range(k)}}
        for k in (2, 3, 4, 6)
    }
    questions["s"] = {"type": "score", "instructions": "How much?", "criteria": ["low", "mid", "high", "max"]}
    questions["n"] = {"type": "noul", "instructions": "Yes?", "criteria": {"true": "it is", "false": "it isn't"}}
    status, body = _call(f"{server}/v1/systemone", {"state": "s", "questions": questions})
    assert status == 200
    for k in (2, 3, 4, 6):
        probs = body["answers"][f"c{k}"]["probabilities"]
        assert list(probs) == [f"o{i}" for i in range(k)] and abs(sum(probs.values()) - 1) < 1e-3
    assert set(body["answers"]["s"]["probabilities"]) == {"0", "1", "2", "3"}
    assert 0 <= body["answers"]["n"]["noul"] <= 1


def test_an_inference_failure_answers_500_not_silence() -> None:
    class Broken(Fake):
        def read(self, record: Record) -> tuple[list[torch.Tensor], int]:
            raise RuntimeError("There is no Stream(gpu, 1) in current thread.")

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler(Broken(), "den-test", None))  # type: ignore[arg-type]
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        status, body = _call(f"http://127.0.0.1:{httpd.server_address[1]}/v1/systemone", KEV_EXAMPLE, key=None)
    finally:
        httpd.shutdown()
    assert status == 500 and "RuntimeError" in body["error"]
