"""`POST /v1/systemone`: a trained run behind Kev's (and TypeSafe's) System One API, so existing clients work unchanged.

    den serve --run runs/kev-recipe/4-skills          # or --run hf:<org>/<name>; listens on 127.0.0.1:8009
    curl -s localhost:8009/v1/systemone -H 'content-type: application/json' -d '{"state": "...", "questions": {...}}'

The response matches Kev's: per question, `choice` (the most likely option), `score` (the expected level) or `noul`
(the probability of true), with `confidence` = (p_max - 1/K) / (1 - 1/K) and every option's probability. Each question
is read with the state alone. Set DEN_API_KEY to require `Authorization: Bearer <key>` on /v1/*. Standard library only.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

from .api import Json, Question, RecordError
from .evaluate import Model, request
from .release import locate

DEFAULT_NAME = "den-latest"


def answer(q: Question, p: list[float]) -> dict[str, Any]:
    """One question's answer in Kev's response shape."""
    if q.type == "noul":
        return {"type": "noul", "noul": round(p[1], 4)}
    k, top = len(p), max(range(len(p)), key=p.__getitem__)
    confidence = round((p[top] - 1 / k) / (1 - 1 / k), 4) if k > 1 else 1.0
    probabilities = {key: round(x, 4) for key, x in zip(q.keys, p, strict=True)}
    if q.type == "choice":
        return {"type": "choice", "choice": q.keys[top], "confidence": confidence, "probabilities": probabilities}
    return {
        "type": "score",
        "score": round(sum(i * x for i, x in enumerate(p)), 4),
        "confidence": confidence,
        "legend": dict(zip(q.keys, q.options, strict=True)),
        "probabilities": probabilities,
    }


def respond(model: Model, raw: dict[str, Json], name: str = DEFAULT_NAME) -> dict[str, Any]:
    """The /v1/systemone response body for one request."""
    started = time.perf_counter()
    record = request(raw)
    scores, tokens = model.read(record)
    answers = {q.id: answer(q, s.softmax(-1).tolist()) for q, s in zip(record.questions, scores, strict=True)}
    return {
        "model": str(raw.get("model") or name),
        "answers": answers,
        "usage": {"input_tokens": tokens, "output_tokens": 0},
        "latency_ms": round(1000 * (time.perf_counter() - started)),
    }


def handler(model: Model, name: str, key: str | None) -> type[BaseHTTPRequestHandler]:
    lock = threading.Lock()  # one forward pass at a time: the backbones are not thread-safe

    class Handler(BaseHTTPRequestHandler):
        server_version = "den"

        def _send(self, status: HTTPStatus, body: dict[str, Any]) -> None:
            data = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _authorized(self) -> bool:
            if key is None or not self.path.startswith("/v1/"):
                return True
            if self.headers.get("authorization", "") == f"Bearer {key}":
                return True
            self._send(HTTPStatus.UNAUTHORIZED, {"error": "missing or wrong bearer key"})
            return False

        def do_GET(self) -> None:
            if not self._authorized():
                return
            if self.path == "/health":
                self._send(HTTPStatus.OK, {"status": "ok"})
            elif self.path == "/v1/models":
                device = model.backbone.device
                self._send(HTTPStatus.OK, {"data": [{"id": name, "backend": device.backend, "dtype": device.dtype,
                                                     "temperature": model.temperature}]})  # fmt: skip
            else:
                self._send(HTTPStatus.NOT_FOUND, {"error": f"no route {self.path}"})

        def do_POST(self) -> None:
            if not self._authorized():
                return
            if self.path != "/v1/systemone":
                self._send(HTTPStatus.NOT_FOUND, {"error": f"no route {self.path}"})
                return
            try:
                raw = json.loads(self.rfile.read(int(self.headers.get("content-length", 0))) or b"null")
                if not isinstance(raw, dict):
                    raise RecordError("the body must be a JSON object")
                with lock:
                    body = respond(model, raw, name)
            except (json.JSONDecodeError, RecordError, SystemExit) as e:
                self._send(HTTPStatus.BAD_REQUEST, {"error": str(e)})
                return
            except Exception as e:  # never drop the connection: say what failed
                self._send(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": f"{type(e).__name__}: {e}"})
                return
            self._send(HTTPStatus.OK, body)

        def log_message(self, format: str, *args: object) -> None:
            print(f"{self.address_string()} {format % args}", flush=True)

    return Handler


def serve_main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="den serve")
    p.add_argument("--run", required=True, help="a run directory, or hf:<org>/<name>[@<revision>]")
    p.add_argument("--host", default="127.0.0.1", help="0.0.0.0 to accept other machines")
    p.add_argument("--port", type=int, default=8009)
    p.add_argument("--name", default=DEFAULT_NAME, help="the model id this server reports")
    p.add_argument("--backend")
    args = p.parse_args(argv)
    model = Model(locate(args.run), args.backend)
    # One thread: inference runs on the thread that loaded the model. MLX's GPU stream belongs to that thread (a
    # request on another one fails with "no Stream(gpu) in current thread"), and requests are serialized anyway.
    server = HTTPServer((args.host, args.port), handler(model, args.name, os.environ.get("DEN_API_KEY")))
    print(f"den serving {args.run} as {args.name} on http://{args.host}:{args.port}/v1/systemone", flush=True)
    with contextlib.suppress(KeyboardInterrupt):
        server.serve_forever()
    return 0
