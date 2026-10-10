"""`den serve`: a trained run behind Kev's `POST /v1/systemone`, so existing clients work unchanged; and
`den predict`, the same responses for request lines. Standard library only.

    den serve --run runs/round2               # or --run hf:<org>/<name>; listens on 127.0.0.1:8009
    curl -s localhost:8009/v1/systemone -H 'content-type: application/json' -d '{"state": "...", "questions": {...}}'
    den predict --run runs/round2 requests.jsonl

Set DEN_API_KEY to require `Authorization: Bearer <key>` on /v1/*.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

from .api import RecordError
from .paths import locate
from .runtime import DEFAULT_NAME, Model, respond


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
    # one thread: MLX's GPU stream belongs to the thread that loaded the model
    server = HTTPServer((args.host, args.port), handler(model, args.name, os.environ.get("DEN_API_KEY")))
    print(f"den serving {args.run} as {args.name} on http://{args.host}:{args.port}/v1/systemone", flush=True)
    with contextlib.suppress(KeyboardInterrupt):
        server.serve_forever()
    return 0


def predict_main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="den predict")
    p.add_argument("--run", required=True, help="a run directory, or hf:<org>/<name>[@<revision>]")
    p.add_argument("requests", nargs="?", default="-", help="JSONL of /v1/systemone requests; - reads stdin")
    p.add_argument("--backend")
    args = p.parse_args(argv)
    model = Model(locate(args.run), args.backend)
    text = sys.stdin.read() if args.requests == "-" else Path(args.requests).read_text(encoding="utf-8")
    for line in filter(str.strip, text.splitlines()):
        print(json.dumps(respond(model, json.loads(line))), flush=True)
    return 0
