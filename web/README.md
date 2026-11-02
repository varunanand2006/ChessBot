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
| `index.html` | The whole UI — board, evaluation bar, undo/redo, move list, strength slider, FEN loader with preset positions, side picker (White / Black / engine vs engine), and a tablebase panel. One file, no framework. |
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

## Loading positions and demoing the tablebases

Paste a FEN into the box under the board (missing trailing fields are filled in)
or pick a preset. You take the side to move; switch **You play** to Black to
defend instead, or to **engine vs engine** and press Play to watch it play out.

With `CHESS_TB_DIR` set, the tablebase panel shows the exact result for the
current position, and the line under it shows how many nodes the engine searched
for its last move — a handful in a tablebase position, versus millions for the
heuristic search. Tablebase mates can run past 50 moves without a capture, so
untick **Enforce the 50-move rule** to watch a deep mate through to the end (the
5-piece presets whose mates are that long untick it for you).

To find demo positions, run `chess tbload <dir>/<MATERIAL>.tb`: it prints the
deepest mate in that table with White to move and with Black to move. Add any you
want as presets in the `PRESETS` list at the top of the script in `index.html`.
