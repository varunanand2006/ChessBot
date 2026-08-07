#!/usr/bin/env python3
# External verification of our generated endgame tables against the Lichess
# tablebase API (which serves true depth-to-mate from Gaviota for <=7 pieces).
#
# For each material we ask the C++ engine for a sample of positions with our
# win/draw/loss + signed distance-to-mate ("chess tbdump"), then query Lichess
# for the same FENs and compare. A match on category AND signed DTM across many
# positions from an independent source is strong evidence our retrograde
# generation is correct — not just self-consistent between our two solvers.
#
# Convention (verified against the API): signed DTM is plies to mate, positive
# if the side to move is winning, negative if being mated, 0 for a draw. Draws
# report dtm=null on Lichess, so only the category is compared there.
#
# Usage (from the repo root):
#   python python/verify_tablebase.py [N] [MATERIAL ...]
#   e.g.  python python/verify_tablebase.py 25 KQK KRK KQKR
#
# Requires network access. Not part of the offline ctest suite.

import json
import os
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHESS = os.path.join(REPO, "build", "chess.exe")
API = "https://tablebase.lichess.ovh/standard"
USER_AGENT = "chess-tb-verify (portfolio project)"
DELAY_S = 0.25  # be polite to the public API


def dump(material, n):
    out = subprocess.check_output([CHESS, "tbdump", material, str(n)], text=True)
    rows = []
    for line in out.strip().splitlines():
        fen, cat, dtm = line.rsplit(",", 2)
        rows.append((fen, cat, int(dtm)))
    return rows


def query(fen):
    # Shell out to curl: it carries its own CA bundle and avoids Python's SSL
    # trust-store issues in some environments. -G + --data-urlencode encodes the
    # FEN (spaces, slashes) safely.
    out = subprocess.check_output(
        ["curl", "-s", "--fail", "--max-time", "20", "-A", USER_AGENT,
         "-G", API, "--data-urlencode", "fen=" + fen],
        text=True)
    return json.loads(out)


def lichess_category(j):
    c = j.get("category", "unknown")
    if c in ("win", "cursed-win", "maybe-win"):
        return "win"
    if c in ("loss", "blessed-loss", "maybe-loss"):
        return "loss"
    return c  # draw / unknown


def dtm_ok(ours, theirs, category):
    # Draws: Lichess reports dtm=null -> compare category only.
    if category == "draw":
        return True
    # Already-mated positions (our dtm 0) may come back as 0 or null.
    if ours == 0:
        return theirs in (0, None)
    return theirs == ours


def main():
    args = sys.argv[1:]
    n = 25
    materials = ["KQK", "KRK", "KQKR"]
    if args and args[0].isdigit():
        n = int(args[0])
        args = args[1:]
    if args:
        materials = args

    if not os.path.exists(CHESS):
        print(f"engine not found at {CHESS} (build first)")
        return 1

    total, mism = 0, 0
    for material in materials:
        print(f"\n=== {material} (sampling {n}) ===")
        rows = dump(material, n)
        m_mism = 0
        for fen, our_cat, our_dtm in rows:
            try:
                j = query(fen)
            except Exception as e:
                print(f"  [ERR ] {fen}  API error: {e}")
                m_mism += 1
                total += 1
                continue
            their_cat = lichess_category(j)
            their_dtm = j.get("dtm")
            ok = (our_cat == their_cat) and dtm_ok(our_dtm, their_dtm, our_cat)
            total += 1
            if not ok:
                m_mism += 1
                print(f"  [FAIL] {fen}")
                print(f"         ours: {our_cat} dtm {our_dtm:+d}   "
                      f"lichess: {their_cat} dtm {their_dtm}")
            time.sleep(DELAY_S)
        mism += m_mism
        status = "OK" if m_mism == 0 else f"{m_mism} MISMATCH"
        print(f"  {material}: {len(rows) - m_mism}/{len(rows)} match  [{status}]")

    print(f"\n==== {total - mism}/{total} positions match the Lichess tablebase ====")
    return 0 if mism == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
