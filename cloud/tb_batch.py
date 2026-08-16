#!/usr/bin/env python3
"""
tb_batch.py -- idempotent tablebase generation worker.

For each material in the work list, if its <MATERIAL>.tb is not already present
in the destination, solve it with the chess binary and upload it. Skipping
already-present tables is what makes the worker:

  * idempotent  -- safe to re-run; a finished table is never recomputed.
  * resumable   -- a spot-instance interruption just leaves that one material
                   undone; the next run (or another worker) picks it up.
  * horizontally scalable -- run N copies against one shared destination and each
                   grabs whatever isn't done yet. Two workers may briefly race on
                   the same material, wasting one solve, but never produce a wrong
                   table (the output is identical either way). That "skip what's
                   already in the bucket" idempotency is why no lock/queue service
                   is needed for a first version.

The destination is a local directory (for testing) or an s3://bucket/prefix (the
real run); the same code drives both, so this script is validated locally and
then runs unchanged on the cloud GPU instances.

Materials must use the canonical name the engine probes by: K + White's pieces
(Q,R,B,N order) + K + Black's pieces, e.g. KQKR, KRKN, KQRKR.

Usage:
    tb_batch.py --chess ./build/chess.exe --dest ./bucket KQKR KRKN
    tb_batch.py --chess ./chess --dest s3://my-tb-bucket/v1 --manifest materials.txt
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile


def dest_has(dest, name):
    """True if <name>.tb already exists at the destination."""
    key = name + ".tb"
    if dest.startswith("s3://"):
        r = subprocess.run(["aws", "s3", "ls", dest.rstrip("/") + "/" + key],
                           capture_output=True, text=True)
        return r.returncode == 0 and key in r.stdout
    return os.path.exists(os.path.join(dest, key))


def dest_put(dest, local_path, name):
    """Upload local_path as <name>.tb to the destination."""
    key = name + ".tb"
    if dest.startswith("s3://"):
        subprocess.run(["aws", "s3", "cp", local_path, dest.rstrip("/") + "/" + key],
                       check=True)
    else:
        os.makedirs(dest, exist_ok=True)
        shutil.copy2(local_path, os.path.join(dest, key))


def solve(chess, material, local_path):
    """Run `chess tbsave <material> <local_path>`; nonzero exit raises."""
    subprocess.run([chess, "tbsave", material, local_path], check=True)


def main():
    ap = argparse.ArgumentParser(description="Idempotent tablebase generation worker.")
    ap.add_argument("--chess", required=True, help="path to the chess binary")
    ap.add_argument("--dest", required=True, help="output directory or s3://bucket/prefix")
    ap.add_argument("--manifest", help="file with one material per line (# comments ok)")
    ap.add_argument("materials", nargs="*", help="materials, e.g. KQKR KRKN")
    args = ap.parse_args()

    work = list(args.materials)
    if args.manifest:
        with open(args.manifest) as f:
            work += [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]
    if not work:
        print("no materials given (pass some, or --manifest)", file=sys.stderr)
        return 2

    generated = skipped = 0
    for m in work:
        if dest_has(args.dest, m):
            print("[skip]  %s already present in %s" % (m, args.dest))
            skipped += 1
            continue
        with tempfile.TemporaryDirectory() as td:
            local_path = os.path.join(td, m + ".tb")
            print("[solve] %s ..." % m)
            solve(args.chess, m, local_path)
            dest_put(args.dest, local_path, m)
            print("[done]  %s -> %s" % (m, args.dest))
            generated += 1

    print("\nsummary: %d generated, %d skipped, %d total"
          % (generated, skipped, len(work)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
