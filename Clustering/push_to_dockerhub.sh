#!/usr/bin/env bash
set -e

echo "========================================================"
echo "   Docker Hub Build and Push Tool - Clustering App"
echo "========================================================"

DOCKER_USER="$1"
if [ -z "$DOCKER_USER" ]; then
    read -p "Enter your Docker Hub username: " DOCKER_USER
fi

if [ -z "$DOCKER_USER" ]; then
    echo "[ERROR] Docker Hub username cannot be empty."
    exit 1
fi

IMAGE_NAME="${DOCKER_USER}/clustering-app:latest"

echo ""
echo "[1/3] Logging into Docker Hub..."
docker login

echo ""
echo "[2/3] Building image ${IMAGE_NAME}..."
docker build -t "${IMAGE_NAME}" .

echo ""
echo "[3/3] Pushing to Docker Hub..."
docker push "${IMAGE_NAME}"

echo ""
echo "========================================================"
echo "[SUCCESS] Image successfully pushed to Docker Hub!"
echo "Image: ${IMAGE_NAME}"
echo ""
echo "To run anywhere:"
echo "  docker pull ${IMAGE_NAME}"
echo "  docker run -p 8501:8501 ${IMAGE_NAME}"
echo "========================================================"

