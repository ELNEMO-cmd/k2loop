"""Tiny HTTP wrapper around Laya (stdlib only + laya). Runs in its own CPU venv.

POST /predict {"state": ..., "questions": {...}} -> Laya predict() result
GET  /health
"""
from __future__ import annotations

import argparse
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--model", default="convaiinnovations/laya-typed-decisions")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    import laya  # imported here so --help works without torch

    agent = laya.load(args.model, device=args.device)
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, obj: object) -> None:
            body = json.dumps(obj, default=str).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            if self.path.startswith("/health"):
                self._send(200, {"ok": True, "model": args.model, "device": args.device})
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self) -> None:  # noqa: N802
            if not self.path.startswith("/predict"):
                self._send(404, {"error": "not found"})
                return
            try:
                n = int(self.headers.get("Content-Length", "0"))
                req = json.loads(self.rfile.read(n).decode() or "{}")
                with lock:
                    res = agent.predict(req["state"], req["questions"])
                self._send(200, res)
            except Exception as exc:  # noqa: BLE001
                self._send(400, {"error": f"{type(exc).__name__}: {exc}"})

        def log_message(self, *a: object) -> None:  # quiet
            return

    print(f"laya server on {args.host}:{args.port} model={args.model} device={args.device}", flush=True)
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
