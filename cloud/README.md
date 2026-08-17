# cloud/ — generating tablebases in the cloud

Generate endgame tablebases on rented GPUs and store them in S3, all provisioned
as code. Download a `.tb` file back, point `CHESS_TB_DIR` at it, and the engine
probes it for perfect play.

There are two provisioning targets, sharing one worker:

| Target | Compute | Used for |
|---|---|---|
| `terraform-runpod/` | **Multi-GPU RunPod pod** + an AWS S3 bucket | The real run — all 28 five-piece tables on a 4-GPU pod, in under an hour |
| `terraform/` | A single AWS spot GPU instance (`g5.xlarge`, A10G), all-AWS | A fully-automated, self-terminating single-GPU alternative |

Both run the same idempotent worker (`tb_batch.py`) and the same fan-out
(`fanout.sh`); only the provisioning differs.

## The multi-GPU model

The parallelism is embarrassingly simple, which is the point: there is no
inter-GPU communication. `fanout.sh` reads the GPU count from `nvidia-smi`, gives
each GPU a disjoint round-robin slice of the material list, and starts one
`tb_batch.py` worker per GPU pinned to its device. N GPUs finish in ~1/N the time.

```
              materials: KQRKR KRBKN KBNKN KQBKR KRNKB ...
fanout.sh  ──▶  GPU 0: KQRKR KRNKB ...   ─┐
                GPU 1: KRBKN ...          ─┤  each worker: solve on its GPU,
                GPU 2: KBNKN ...          ─┼─▶  upload <MATERIAL>.tb to S3
                GPU 3: KQBKR ...          ─┘
```

`tb_batch.py` is idempotent — it skips any material already at the destination —
so a reclaimed spot GPU, a re-run, or two workers racing the same material are all
safe: at worst one wasted solve, never a wrong table. That's what makes running on
cheap interruptible instances reasonable.

## Contents

| Path | What it is |
|---|---|
| `fanout.sh` | Fans one worker per GPU across a disjoint slice of the material list. |
| `tb_batch.py` | The worker: for each material, skip if already present, else solve and upload. 5-piece → GPU sweep, ≤4-piece → CPU. Local dir or `s3://` destination. |
| `Dockerfile` | Multi-stage worker image. CUDA `devel` compiles `chess` + `cuda_sweep_check`; `runtime` ships them + the AWS CLI. |
| `terraform-runpod/` | RunPod GPU pod + an AWS S3 bucket and a scoped IAM user whose key is injected into the pod. |
| `terraform/` | The all-AWS alternative: S3 bucket, least-privilege instance role, spot `g5.xlarge` that builds, runs, and self-terminates. |

Each `terraform*/` directory has a `tf.sh` that runs Terraform inside a container
(the standard CI pattern; here it also sidesteps a local TLS-inspection driver
that breaks native Terraform's core-to-plugin TLS).

## Run the worker locally (CPU, no GPU)

The same script drives local and cloud runs, so it's validated on a laptop first.
`≤4-piece` materials solve on the CPU in seconds:

```bash
python cloud/tb_batch.py --chess ./build/chess.exe --dest ./bucket KRK KQKR
```

Re-running skips what's already there. Then probe the result:

```bash
CHESS_TB_DIR=./bucket ./build/chess.exe search 12 "8/8/8/8/8/2k5/8/R3K3 w - - 0 1"
```

## Run on multiple GPUs (RunPod → S3)

From `terraform-runpod/`, with AWS credentials configured and a `RUNPOD_API_KEY`
set:

```bash
cp terraform.tfvars.example terraform.tfvars   # set gpu_count, materials, region
./tf.sh init
./tf.sh apply     # creates the S3 bucket + scoped IAM user + the GPU pod
```

One `apply` creates the bucket, a bucket-scoped IAM user (the pod isn't an AWS
principal, so it uploads with a scoped access key rather than an instance role),
and the pod — with `DEST`, the material list, and the AWS key already injected
into the pod's environment.

The pod boots to an idle container rather than running the batch as its start
command: RunPod expects a long-lived container and restarts a start command that
exits, so a finite batch would crash-loop with no visibility. Instead, SSH in (or
use the RunPod web terminal) and run the build-and-fan-out steps — they're in
[`terraform-runpod/pod_start.sh`](terraform-runpod/pod_start.sh), and every value
they read is already in the environment. When the batch prints `BATCH COMPLETE`,
tear the pod down (pods don't self-terminate):

```bash
./tf.sh destroy                                  # removes the pod
./tf.sh destroy -target=runpod_pod.batch         # ...or just the pod, keep the bucket
```

## Run on a single AWS GPU (fully automated)

From `terraform/`, the instance's `user_data` builds the image, generates the
materials, uploads them, and terminates itself — no manual step. Watch it over SSM
(no SSH, no inbound ports):

```bash
cp terraform.tfvars.example terraform.tfvars
./tf.sh init && ./tf.sh apply
aws ssm start-session --target <instance-id>     # instance-id is a terraform output
tail -f /var/log/cloud-init-output.log
```

`./tf.sh destroy` removes the bucket and instance; pin `bucket_name` in
`terraform.tfvars` to keep the tables past teardown.

## Two things worth knowing

**The image is built on the instance/pod, not pulled.** The engine compiles with
`-march=native`, so building on the box that runs it targets that exact CPU;
building elsewhere risks a SIGILL on a different ISA.

**Credentials are scoped and short-lived by design.** The pod's IAM user can only
`PutObject`/`ListBucket` on the one tablebase bucket. The all-AWS path avoids a
long-lived key entirely with an instance role; the RunPod path needs a key
(off-AWS compute can't assume a role), so it's scoped to that single bucket and
destroyed with the rest.

## Scope

The worker handles distinct-piece pawnless materials up to 5 pieces (the set the
combinatorial indexer and GPU sweep cover). Duplicate-piece materials (KRR vs. K)
and pawns are out of scope; see the roadmap in the top-level `CLAUDE.md`.
