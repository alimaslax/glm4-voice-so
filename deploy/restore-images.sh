#!/usr/bin/env bash
# Restore the exact Docker image the fast GLM-4-Voice stack runs on (torch 2.9.1+cu128, bitsandbytes, Blackwell-ready).
# Run once on a fresh GPU VM after mounting the network volume at /workspace. Needs docker + nvidia-container-toolkit.
set -euo pipefail
cd /workspace/migration

IMG=glm4voice-runtime-blackwell:0.2
TAR=docker-image-glm4voice-blackwell.tar.zst

if docker image inspect "$IMG" >/dev/null 2>&1; then
  echo "$IMG already present."; exit 0
fi

if [ -f "$TAR" ]; then
  echo "Checking $TAR ..."
  sha256sum -c "$TAR.sha256"
  echo "Loading $IMG (self-contained, ~10 GB) ..."
  zstd -dc "$TAR" | docker load
  docker image inspect "$IMG" >/dev/null && echo "OK: $IMG restored."
  exit 0
fi

echo "No $TAR found; falling back to legacy restore + rebuild of the Blackwell layers."
./restore-images.legacy.sh
docker build -f docker-blackwell/Dockerfile     -t glm4voice-runtime-blackwell:0.1 docker-blackwell
docker build -f docker-blackwell/Dockerfile.fix -t "$IMG" docker-blackwell
