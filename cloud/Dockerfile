# Tablebase generation worker image.
#
# This is the container each cloud GPU instance runs: it builds the engine and
# the CUDA retrograde-sweep binary, then runs the idempotent batch worker
# (cloud/tb_batch.py) to solve a list of materials and upload the .tb files to
# object storage.
#
# Multi-stage on purpose:
#   * builder  = the CUDA *devel* image (ships nvcc + headers) to compile.
#   * runtime  = the CUDA *runtime* image (cudart only, no compiler) that just
#                carries the built binaries + the Python worker. The devel image
#                is ~2x the size; shipping only the runtime layer keeps what the
#                spot instance pulls small (a real cost when N workers each pull
#                on boot).
#
# CUDA 12.6 / Ubuntu 24.04 gives gcc-13 as nvcc's host compiler — close to the
# project's gcc-16 dev toolchain and comfortably C++20-complete. (The dev box is
# MinGW gcc-16; here we build the same C++20 sources under Linux gcc-13.)

# ---------------------------------------------------------------------------
# Stage 1: build
# ---------------------------------------------------------------------------
FROM nvidia/cuda:12.6.2-devel-ubuntu24.04 AS builder

# Build-only deps. No git: the source is COPYd from the build context, so the
# image never reaches out to a remote during build.
RUN apt-get update && apt-get install -y --no-install-recommends \
        cmake ninja-build build-essential ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/chess
COPY . .

# CMAKE_CUDA_ARCHITECTURES MUST be set explicitly here. The project's CMake
# defaults it to `native`, which asks CMake to detect the *installed GPU's*
# arch — but there is no GPU during `docker build`, so `native` would fail to
# configure. We target both cards this project runs on:
#   86 = Ampere sm_86  -> AWS g5 (A10G)
#   89 = Ada    sm_89  -> RTX 4090 (RunPod backup)
# One fatbin covers both, so the same image runs on either without a rebuild.
#
# Only `chess` and `cuda_sweep_check` are built — the two binaries the worker
# needs. The tests/benchmarks still configure (add_subdirectory) but are not
# compiled, cutting image build time. To run the on-box bit-exact gate suite,
# build with no --target (plain `cmake --build build`) and `ctest` on a GPU host.
#
# NOTE on -march=native: the project's Release flags bake in the *build host's*
# CPU ISA. Build this image on the instance family it will run on (the Terraform
# user_data does exactly that), or on a matching x86-64-v3 host — otherwise a
# newer build CPU can emit instructions the runtime CPU faults on (SIGILL).
RUN cmake -S . -B build -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CUDA_ARCHITECTURES="86;89" \
    && cmake --build build --target chess cuda_sweep_check

# ---------------------------------------------------------------------------
# Stage 2: runtime
# ---------------------------------------------------------------------------
FROM nvidia/cuda:12.6.2-runtime-ubuntu24.04 AS runtime

# Runtime deps: python3 to drive the worker, the AWS CLI for S3 put/get, and CA
# certs for its TLS calls. Ubuntu 24.04 (noble) dropped the `awscli` apt package,
# so we install the official AWS CLI v2 bundle — a self-contained install that
# needs no pip (avoiding noble's PEP 668 externally-managed restriction). curl +
# unzip are build-only helpers, purged afterward to keep the layer small.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 curl unzip ca-certificates \
    && curl -fsSL "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o /tmp/awscliv2.zip \
    && unzip -q /tmp/awscliv2.zip -d /tmp \
    && /tmp/aws/install \
    && rm -rf /tmp/awscliv2.zip /tmp/aws \
    && apt-get purge -y --auto-remove curl unzip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/chess

# Copy just the artifacts the worker invokes: the two binaries and the script.
COPY --from=builder /opt/chess/build/chess                 /opt/chess/build/chess
COPY --from=builder /opt/chess/build/cuda/cuda_sweep_check /opt/chess/build/cuda/cuda_sweep_check
COPY --from=builder /opt/chess/cloud/tb_batch.py           /opt/chess/cloud/tb_batch.py

# The image *is* the worker: `docker run --gpus all <img> --dest s3://bucket/prefix KQRKR ...`.
# --chess and --sweep are baked so callers only supply the destination + materials.
# The worker routes 5-man+ to cuda_sweep_check (GPU, needs --gpus all) and <=4-man
# to `chess tbsave` (CPU); each is the faster correct path for its size.
ENTRYPOINT ["python3", "/opt/chess/cloud/tb_batch.py", \
            "--chess", "/opt/chess/build/chess", \
            "--sweep", "/opt/chess/build/cuda/cuda_sweep_check"]
CMD ["--help"]
