#!/usr/bin/env bash
# fanout.sh -- run the tablebase batch across every GPU on the box.
#
# The multi-GPU story is the "batch" axis: each GPU gets a disjoint slice of the
# material list and runs its own tb_batch.py worker, so N GPUs give ~N x the
# throughput with no inter-GPU communication. This is what runs on the RunPod
# multi-GPU pod; it works on any multi-GPU Linux box (nvidia-smi + the built
# binaries) so it can be smoke-tested outside RunPod too.
#
# tb_batch is idempotent (skips a table already at --dest), so a crashed GPU's
# slice, a re-run, or even two workers racing the same material are all safe --
# never a wrong table, at worst one wasted solve. The round-robin split already
# keeps the slices disjoint, so there is no overlap in normal operation.
#
# Config via env (with sensible defaults) so the same script drops into a RunPod
# start command or an SSH session unchanged:
#   DEST      output dir or s3://bucket/prefix   (required)
#   CHESS     path to the chess binary           (default build/chess)
#   SWEEP     path to cuda_sweep_check           (default build/cuda/cuda_sweep_check)
#   MANIFEST  optional file of materials, one per line (# comments ok)
# Materials may also be passed as positional args. Example:
#   DEST=s3://my-tb-bucket/v1 ./cloud/fanout.sh KQRKR KRBKN KBNKN KQBKR
set -euo pipefail

CHESS="${CHESS:-build/chess}"
SWEEP="${SWEEP:-build/cuda/cuda_sweep_check}"
DEST="${DEST:?set DEST to a directory or s3://bucket/prefix}"

# GPU count from nvidia-smi. (If the caller pins CUDA_VISIBLE_DEVICES to a subset
# before calling, unset it so the per-GPU children below control their own device.)
NGPU="$(nvidia-smi -L | wc -l)"
[ "$NGPU" -ge 1 ] || { echo "fanout: no GPUs found (nvidia-smi -L empty)"; exit 1; }
unset CUDA_VISIBLE_DEVICES || true

# Collect materials from args + optional manifest.
mats=("$@")
if [ -n "${MANIFEST:-}" ]; then
  while IFS= read -r m; do
    [ -n "$m" ] && [[ "$m" != \#* ]] && mats+=("$m")
  done < "$MANIFEST"
fi
[ "${#mats[@]}" -ge 1 ] || { echo "fanout: no materials (pass args or set MANIFEST)"; exit 2; }

echo "fanout: ${#mats[@]} materials across $NGPU GPU(s) -> $DEST"

pids=()
for ((g = 0; g < NGPU; g++)); do
  # Round-robin slice for GPU g: material indices g, g+NGPU, g+2*NGPU, ...
  slice=()
  for ((i = g; i < ${#mats[@]}; i += NGPU)); do slice+=("${mats[$i]}"); done
  [ "${#slice[@]}" -ge 1 ] || continue  # more GPUs than materials: this one idles
  echo "  GPU $g <- ${slice[*]}"
  CUDA_VISIBLE_DEVICES="$g" python3 cloud/tb_batch.py \
    --chess "$CHESS" --sweep "$SWEEP" --dest "$DEST" "${slice[@]}" \
    > "fanout_gpu$g.log" 2>&1 &
  pids+=($!)
done

# Wait for every worker; the run fails if any slice failed.
rc=0
for p in "${pids[@]}"; do wait "$p" || rc=1; done
echo "fanout: done (rc=$rc); per-GPU logs in fanout_gpu*.log"
exit "$rc"
