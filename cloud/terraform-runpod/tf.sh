#!/usr/bin/env bash
# tf.sh -- run Terraform in the official hashicorp/terraform container.
#
# Why a container instead of native terraform: this machine has a TLS-inspection
# filter driver that MITMs even loopback traffic, which breaks terraform-core's
# mutual-TLS handshake to its provider plugins ("x509: certificate signed by
# unknown authority"). Inside a Linux container the plugin loopback lives in the
# container's own network namespace, so the host filter never sees it. Running
# terraform from this image is also how most CI/CD pipelines run it, so this is a
# standard pattern, not a workaround that bites later.
#
# Norton's SSL/TLS-scanning driver re-signs HTTPS (including the container's calls
# to registry.terraform.io) with its own CA, which the Linux container does not
# trust by default. If cloud/win-roots.pem exists (the exported Windows trust
# store, which includes Norton's CA), we mount it and point SSL_CERT_FILE at it so
# terraform trusts exactly what Windows already does. Regenerate it per machine:
#   powershell: export Cert:\LocalMachine\Root + CurrentUser\Root to cloud/win-roots.pem
#
# Usage:
#   ./tf.sh init            # first time, or after changing providers/backend
#   ./tf.sh validate
#   ./tf.sh plan            # first step that hits AWS (needs creds); costs $0
#   ./tf.sh apply           # creates billable infra
#   ./tf.sh destroy
#
# Credentials: mounts ~/.aws read-only. For a named or SSO profile, export
# AWS_PROFILE first (and `aws sso login` on the host so the cached token is live).
set -euo pipefail

cd "$(dirname "$0")"
# Windows-form path (C:/...) so Docker Desktop resolves the bind mount; plain pwd
# elsewhere. MSYS_NO_PATHCONV below stops Git Bash rewriting the container paths.
WORKDIR="$(pwd -W 2>/dev/null || pwd)"
IMAGE="${TF_IMAGE:-hashicorp/terraform:latest}"

args=(--rm
      -v "$WORKDIR":/work -w /work
      -e AWS_PROFILE -e AWS_REGION -e AWS_DEFAULT_REGION -e RUNPOD_API_KEY)

# Mount host AWS creds when present: needed for plan/apply, harmless for init.
if [ -d "$HOME/.aws" ]; then
  AWSDIR="$(cd "$HOME/.aws" && { pwd -W 2>/dev/null || pwd; })"
  args+=(-v "$AWSDIR":/root/.aws:ro)
fi

# Trust the Windows CA bundle (incl. Norton's SSL-scanning CA) if it's been
# exported, so registry/API TLS verifies through Norton's MITM. Lives one level
# up (cloud/win-roots.pem), shared by every terraform config dir.
if [ -f "../win-roots.pem" ]; then
  CA_HOST="$(cd .. && { pwd -W 2>/dev/null || pwd; })/win-roots.pem"
  args+=(-v "$CA_HOST":/etc/tf-ca.pem:ro -e SSL_CERT_FILE=/etc/tf-ca.pem)
fi

# Attach a TTY only when there is one, so `apply` can prompt interactively while
# a non-TTY run (CI, this repo's tooling) still works.
[ -t 0 ] && args+=(-it)

export MSYS_NO_PATHCONV=1
exec docker run "${args[@]}" "$IMAGE" "$@"
