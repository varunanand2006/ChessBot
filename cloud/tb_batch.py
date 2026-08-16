#!/usr/bin/env python3
"""
tb_batch.py -- idempotent tablebase generation worker.

For each material in the work list, if its <MATERIAL>.tb is not already present
in the destination, solve it and upload it. Skipping already-present tables is
what makes the worker:

  * idempotent  -- safe to re-run; a finished table is never recomputed.
  * resumable   -- a spot-instance interruption just leaves that one material
                   undone; the next run (or another worker) picks it up.
  * horizontally scalable -- run N copies against one shared destination and each
                   grabs whatever isn't done yet. Two workers may briefly race on
                   the same material, wasting one solve, but never produce a wrong
                   table (the output is identical either way). That "skip what's
                   already in the bucket" idempotency is why no lock/queue service
                   is needed for a first version.

Solve backends:
  * CPU (--chess):  `chess tbsave <material> <out>` -- the host sweep. Fast for
                    <=4-man.
  * GPU (--sweep):  `cuda_sweep_check <material>` with CHESS_TB_OUT=<out> -- the
                    retrograde sweep on the device, ~40-60 s/material on an A10G.
                    This is the point of a GPU instance.

Routing: a 5-man (or larger) material goes to the GPU when --sweep is given,
otherwise to the CPU. <=4-man always uses the CPU path -- the GPU harness runs a
slow CPU oracle for those, so tbsave is the faster route there.

The destination is a local directory (for testing) or an s3://bucket/prefix (the
real run); the same code drives both, so this script is validated locally and
then runs unchanged on the cloud GPU instances.

Materials must use the canonical name the engine probes by: K + White's pieces
(Q,R,B,N order) + K + Black's pieces, e.g. KQKR, KRKN, KQRKR.

Usage:
    tb_batch.py --chess ./build/chess.exe --dest ./bucket KRK KQKR
    tb_batch.py --chess /opt/chess/build/chess \
                --sweep /opt/chess/build/cuda/cuda_sweep_check \
                --dest s3://my-tb-bucket/v1 KQRKR KRBKN
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


def men(material):
    """Number of men (pieces) in a canonical material name, e.g. KQKR -> 4."""
    return sum(1 for c in material if c.isalpha())


def solve_cpu(chess, material, out_path):
    """`chess tbsave <material> <out_path>` (host solve); nonzero exit raises."""
    subprocess.run([chess, "tbsave", material, out_path], check=True)


def solve_gpu(sweep, material, out_path):
    """`cuda_sweep_check <material>` with CHESS_TB_OUT set (device solve).

    The harness writes the table to CHESS_TB_OUT only on its success path, so a
    missing output file means the solve did not happen (e.g. no GPU on the box) --
    surfaced as an error rather than uploading nothing. cwd is the output dir so
    the harness's sample-dump CSV lands beside it (and is discarded with the temp
    dir), not wherever the worker was launched.
    """
    env = dict(os.environ, CHESS_TB_OUT=out_path)
    subprocess.run([sweep, material], check=True, env=env,
                   cwd=os.path.dirname(out_path) or ".")
    if not os.path.exists(out_path):
        raise RuntimeError(
            "%s: sweep produced no table (no GPU, or solve failed)" % material)


def solve(material, out_path, chess, sweep):
    """Pick the backend: GPU for 5-man+ when available, else CPU tbsave."""
    if sweep and men(material) >= 5:
        print("[gpu]   %s (%d-man) via %s" % (material, men(material),
                                              os.path.basename(sweep)))
        solve_gpu(sweep, material, out_path)
    else:
        print("[cpu]   %s (%d-man) via tbsave" % (material, men(material)))
        solve_cpu(chess, material, out_path)


def main():
    ap = argparse.ArgumentParser(description="Idempotent tablebase generation worker.")
    ap.add_argument("--chess", required=True, help="path to the chess binary (CPU solve)")
    ap.add_argument("--sweep", help="path to cuda_sweep_check (GPU solve for 5-man+)")
    ap.add_argument("--dest", required=True, help="output directory or s3://bucket/prefix")
    ap.add_argument("--manifest", help="file with one material per line (# comments ok)")
    ap.add_argument("materials", nargs="*", help="materials, e.g. KQKR KQRKR")
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
            solve(m, local_path, args.chess, args.sweep)
            dest_put(args.dest, local_path, m)
            print("[done]  %s -> %s" % (m, args.dest))
            generated += 1

    print("\nsummary: %d generated, %d skipped, %d total"
          % (generated, skipped, len(work)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
