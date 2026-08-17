#!/usr/bin/env python3
"""
server.py -- tiny bridge between the browser frontend and the C++ engine.

Keeps ONE `chess api` process alive (magic tables built once at startup, not per
request) and forwards each browser request to it as a line of the engine's
stdin/stdout JSON protocol. Pure Python standard library -- no Flask, no
dependencies. It serves index.html and exposes a single POST /api endpoint that
relays {cmd, fen, depth} to the engine and returns the engine's JSON verbatim.

Design: the engine is a stateless query service (every request carries the FEN),
so this server holds no game state either -- undo/redo/history live in the page.
A lock serializes access to the one engine pipe so concurrent browser requests
never interleave on it.

Run:   python web/server.py
Env:   CHESS=<path to chess binary>     (default: build/chess[.exe])
       CHESS_TB_DIR=<dir of .tb files>  (optional; lets the bot probe tables)
       PORT=8000
"""
import json
import os
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def find_engine():
    cand = os.environ.get("CHESS")
    if cand and os.path.exists(cand):
        return cand
    for p in ("build/chess.exe", "build/chess"):
        full = os.path.join(ROOT, p)
        if os.path.exists(full):
            return full
    sys.exit("chess binary not found -- build it (cmake --build build) or set CHESS=...")


ENGINE = find_engine()
# CHESS_TB_DIR (if set) is inherited by the engine process, so the bot probes the
# on-disk tablebases exactly as the CLI does.
_proc = subprocess.Popen(
    [ENGINE, "api"],
    stdin=subprocess.PIPE, stdout=subprocess.PIPE,
    text=True, bufsize=1, cwd=ROOT, env=dict(os.environ),
)
_lock = threading.Lock()


def engine_cmd(line):
    """Send one command line; return the single JSON line the engine replies."""
    with _lock:
        _proc.stdin.write(line + "\n")
        _proc.stdin.flush()
        return _proc.stdout.readline().strip()


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype="application/json"):
        data = body.encode("utf-8") if isinstance(body, str) else body
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        path = "index.html" if self.path in ("/", "") else self.path.lstrip("/")
        if ".." in path:  # no directory traversal outside web/
            return self._send(403, "forbidden", "text/plain")
        full = os.path.join(HERE, path)
        if not os.path.isfile(full):
            return self._send(404, "not found", "text/plain")
        ctype = "text/html" if full.endswith(".html") else "text/plain"
        with open(full, "rb") as f:
            self._send(200, f.read(), ctype)

    def do_POST(self):
        if self.path != "/api":
            return self._send(404, "not found", "text/plain")
        n = int(self.headers.get("Content-Length", 0))
        try:
            req = json.loads(self.rfile.read(n) or "{}")
        except ValueError:
            return self._send(400, '{"ok":false,"error":"bad json"}')
        cmd, fen = req.get("cmd"), req.get("fen", "")
        if cmd == "legal":
            line = "legal " + fen
        elif cmd == "go":
            line = "go %d %s" % (int(req.get("depth", 4)), fen)
        else:
            return self._send(400, '{"ok":false,"error":"bad cmd"}')
        self._send(200, engine_cmd(line))

    def log_message(self, *_):  # keep the console quiet
        pass


def main():
    port = int(os.environ.get("PORT", "8000"))
    srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print("engine : %s" % ENGINE)
    print("tables : %s" % os.environ.get("CHESS_TB_DIR", "(none -- set CHESS_TB_DIR to enable)"))
    print("serving: http://127.0.0.1:%d   (Ctrl+C to stop)" % port)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nshutting down")
    finally:
        try:
            _proc.stdin.write("quit\n")
            _proc.stdin.flush()
        except Exception:
            pass


if __name__ == "__main__":
    main()
