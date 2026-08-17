# web/ — browser UI for the engine

A single-page board you can play against the C++ engine in a browser, with no
build step and no dependencies beyond Python's standard library.

```bash
python web/server.py            # serves http://localhost:8000
```

Open **http://localhost:8000** and play (you are White). To let the engine play
perfect endgames, point it at a directory of `.tb` tablebase files:

```bash
CHESS_TB_DIR=/path/to/tables python web/server.py
```

## What's here

| File | What it is |
|---|---|
| `index.html` | The whole UI — board, evaluation bar, undo/redo, move list, strength slider. One file, no framework. |
| `server.py` | ~120-line bridge: keeps one `chess api` process alive and relays browser requests to it over a line protocol. |

## How it fits together

The C++ engine exposes a stateless line protocol (`chess api`): each request
carries a full FEN, so the engine holds no game state. `server.py` starts one
engine process (magic tables built once, not per request), serves `index.html`,
and forwards each browser request as one protocol line — `legal <fen>` for the
position's legal moves and current evaluation, `go <depth> <fen>` for the
engine's move. A lock serializes access to the single engine pipe.

Because the engine is stateless, **all** game state — move history, undo/redo,
the move list — lives in the page. The strength slider simply sets the `depth`
sent with each `go`. Environment variables (`CHESS`, `CHESS_TB_DIR`, `PORT`) are
inherited by the engine process, so tablebase probing works here exactly as it
does from the CLI.
