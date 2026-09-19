#!/usr/bin/env bash
set -euo pipefail

cd /workspace/migration

if [ -f "docker-images.tar.zst" ]; then
    echo "Restoring base runtime images (glm4voice-runtime:0.1, bayling-runtime:latest)..."
    zstd -dc docker-images.tar.zst | docker load
fi

if [ -f "docker-image-glm4voice-int4.tar.zst" ]; then
    echo "Restoring glm4voice-runtime-int4:0.1..."
    zstd -dc docker-image-glm4voice-int4.tar.zst | docker load
fi

echo "Verifying images in docker..."
docker images

echo "All images restored successfully."
