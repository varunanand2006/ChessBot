# cloud/ — tablebase generation on AWS

Generate endgame tablebases on a GPU instance and store them in S3, provisioned
as code. The engine then probes the generated `.tb` files locally.

```
terraform (tf.sh)                 spot g5.xlarge (A10G)               S3
  S3 bucket + IAM + spot   ─▶   git clone ─▶ docker build ─▶   ─▶  <MATERIAL>.tb
  instance (user_data)          run tb_batch.py:                    (versioned,
                                  5-man  → cuda_sweep_check (GPU)     encrypted)
                                  ≤4-man → chess tbsave    (CPU)
                                then shutdown (terminate)
```

Download a `.tb` back to a directory, point `CHESS_TB_DIR` at it, and the search
probes it for exact play (see the tablebase-probing notes in the top-level
`CLAUDE.md`).

## Contents

| Path | What it is |
|---|---|
| `tb_batch.py` | Idempotent worker: for each material, skip if already in the destination, else solve and upload. Local dir or `s3://` destination. |
| `Dockerfile` | Multi-stage worker image. CUDA `devel` compiles `chess` + `cuda_sweep_check` (sm_86/sm_89); `runtime` ships them + the AWS CLI. |
| `terraform/` | S3 bucket, least-privilege IAM role, egress-only security group, spot `g5.xlarge` with `user_data` that builds and runs the worker. |
| `terraform/tf.sh` | Runs Terraform inside a container (see "Terraform runs in a container" below). |

## Run the worker locally (CPU)

The same script drives local and cloud runs, so it is validated on a laptop
first. `<=4-man` materials solve on the CPU in seconds:

```bash
python cloud/tb_batch.py --chess ./build/chess.exe --dest ./bucket KRK KQKR
```

Re-running skips what is already present. Then probe the result:

```bash
CHESS_TB_DIR=./bucket ./build/chess.exe search 12 "8/8/8/8/8/2k5/8/R3K3 w - - 0 1"
```

## Run on AWS

One-time: `aws configure` (creates `~/.aws`). Then, from `cloud/terraform/`:

```bash
cp terraform.tfvars.example terraform.tfvars   # edit region / materials if wanted
./tf.sh init
./tf.sh plan     # first step that hits AWS; costs nothing. Confirms the AMI
                 # resolves and the vCPU quota admits the instance.
./tf.sh apply    # creates the bucket + spot instance, which generates and uploads
```

The instance builds the image, generates the requested materials to
`s3://<bucket>/<prefix>`, and terminates itself. Watch progress over SSM
(no SSH, no inbound ports):

```bash
aws ssm start-session --target <instance-id>   # id is a terraform output
tail -f /var/log/cloud-init-output.log
```

`terraform destroy` removes the bucket and instance. Pin `bucket_name` in
`terraform.tfvars` if you want the tables to outlive the instance.

## Two things worth knowing

**Terraform runs in a container.** `tf.sh` runs the official
`hashicorp/terraform` image with the config directory and `~/.aws` mounted. This
is the standard CI pattern, and here it is also required: a local TLS-inspection
filter driver breaks native Terraform's core-to-plugin mutual TLS, which the
container's own network namespace avoids.

**The image is built on the instance.** `user_data` runs `docker build` on the
box rather than pulling a prebuilt image, so the engine's `-march=native` targets
that instance's CPU. Building elsewhere risks a SIGILL on a different ISA.

## Scope

The worker handles distinct-piece pawnless materials up to 5 men (the set the
combinatorial indexer and GPU sweep cover). Duplicate-piece materials (KRR vs K)
and pawns are out of scope; see the roadmap in the top-level `CLAUDE.md`.
