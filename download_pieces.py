"""
download_pieces.py — run once with any Python to fetch piece PNGs.

Downloads the classic Wikimedia Commons chess piece set (Colin M.L. Burnett,
CC BY-SA 3.0) into a 'pieces/' subfolder next to this script.

Usage:
    python download_pieces.py
"""

import urllib.request
import os
import sys

OUT_DIR = os.path.join(os.path.dirname(__file__), "pieces")
SIZE    = 80   # pixels — change to 60 or 100 if preferred

# Wikimedia Commons SVG-rendered PNG URLs
# Naming: Chess_{piece}{color}t45.svg rendered at {SIZE}px
# color: l = white (light), d = black (dark)
# piece: k q r b n p

BASE = "https://upload.wikimedia.org/wikipedia/commons/thumb"

PIECES = {
    #  local name       Wikimedia path
    "wK": f"{BASE}/4/42/Chess_klt45.svg/{SIZE}px-Chess_klt45.svg.png",
    "wQ": f"{BASE}/1/15/Chess_qlt45.svg/{SIZE}px-Chess_qlt45.svg.png",
    "wR": f"{BASE}/7/72/Chess_rlt45.svg/{SIZE}px-Chess_rlt45.svg.png",
    "wB": f"{BASE}/b/b1/Chess_blt45.svg/{SIZE}px-Chess_blt45.svg.png",
    "wN": f"{BASE}/7/70/Chess_nlt45.svg/{SIZE}px-Chess_nlt45.svg.png",
    "wP": f"{BASE}/4/45/Chess_plt45.svg/{SIZE}px-Chess_plt45.svg.png",
    "bK": f"{BASE}/f/f0/Chess_kdt45.svg/{SIZE}px-Chess_kdt45.svg.png",
    "bQ": f"{BASE}/4/47/Chess_qdt45.svg/{SIZE}px-Chess_qdt45.svg.png",
    "bR": f"{BASE}/f/ff/Chess_rdt45.svg/{SIZE}px-Chess_rdt45.svg.png",
    "bB": f"{BASE}/9/98/Chess_bdt45.svg/{SIZE}px-Chess_bdt45.svg.png",
    "bN": f"{BASE}/e/ef/Chess_ndt45.svg/{SIZE}px-Chess_ndt45.svg.png",
    "bP": f"{BASE}/c/c7/Chess_pdt45.svg/{SIZE}px-Chess_pdt45.svg.png",
}

os.makedirs(OUT_DIR, exist_ok=True)

headers = {"User-Agent": "Mozilla/5.0 chess-gui-downloader/1.0"}

ok = 0
for name, url in PIECES.items():
    dest = os.path.join(OUT_DIR, f"{name}.png")
    if os.path.exists(dest):
        print(f"  skip  {name}.png  (already exists)")
        ok += 1
        continue
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = resp.read()
        with open(dest, "wb") as f:
            f.write(data)
        print(f"  OK    {name}.png  ({len(data)//1024} KB)")
        ok += 1
    except Exception as e:
        print(f"  FAIL  {name}.png  — {e}", file=sys.stderr)

print(f"\n{ok}/12 pieces downloaded to: {OUT_DIR}")
if ok < 12:
    print("Re-run the script to retry failed downloads.")