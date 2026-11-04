#!/usr/bin/env bash
set -e

echo "========================================================"
echo "   Pull and Run Clustering Studio from Docker Hub"
echo "========================================================"

IMAGE_NAME="$1"
if [ -z "$IMAGE_NAME" ]; then
    read -p "Enter full Docker Hub image tag (e.g. yourusername/clustering-app:latest): " IMAGE_NAME
fi

if [ -z "$IMAGE_NAME" ]; then
    echo "[ERROR] Image name cannot be empty."
    exit 1
fi

echo ""
echo "[1/2] Pulling image ${IMAGE_NAME}..."
docker pull "${IMAGE_NAME}"

echo ""
echo "[2/2] Running container on port 8501..."
docker stop clustering-app 2>/dev/null || true
docker rm clustering-app 2>/dev/null || true
docker run -d -p 8501:8501 --name clustering-app "${IMAGE_NAME}"

echo ""
echo "========================================================"
echo "[SUCCESS] Clustering Studio is now running!"
echo "Open in browser: http://localhost:8501"
echo "========================================================"

