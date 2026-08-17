#!/usr/bin/env bash
# Runs once when the RunPod pod boots (passed as docker_start_cmd). Every value
# it needs arrives through the pod's environment (set by the runpod_pod `env`
# map): DEST, MATERIALS, REPO_URL, REPO_REF, and the AWS_* upload credentials.
# The base image is CUDA devel, so nvcc is already present; we add the build
# tools + the AWS CLI, build the engine, then fan the materials across the GPUs.
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive

apt-get update
apt-get install -y --no-install-recommends cmake ninja-build git python3 curl unzip ca-certificates

# AWS CLI v2 bundle (Ubuntu 24.04 dropped the awscli apt package).
curl -fsSL "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o /tmp/aws.zip
unzip -q /tmp/aws.zip -d /tmp
/tmp/aws/install

git clone --depth 1 --branch "$REPO_REF" "$REPO_URL" /workspace/chess
cd /workspace/chess

# CUDA arch = native: on a real GPU the default detects the pod's card (unlike a
# GPU-less docker build, which must pin it). Build only what the worker runs.
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build --target chess cuda_sweep_check

# Fan the materials across every GPU on the pod; each uploads to S3 as it goes.
DEST="$DEST" CHESS=build/chess SWEEP=build/cuda/cuda_sweep_check \
  bash cloud/fanout.sh $MATERIALS

echo "BATCH COMPLETE -> $DEST"
echo "run ./tf.sh destroy to stop the pod (RunPod pods do not self-terminate)"
