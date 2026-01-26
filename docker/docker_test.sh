#!/bin/bash
# Script to build the Docker image and run training tests inside the container

set -e

IMAGE_NAME="slippi-ai"
CONTAINER_NAME="slippi-ai-test"

echo "Building Docker image..."
docker build -t "$IMAGE_NAME" . -f docker/Dockerfile

echo "Running training test in container..."
docker run --rm \
    --gpus all \
    --name "$CONTAINER_NAME" \
    "$IMAGE_NAME" \
    bash tests/training_test.sh

echo "Training test completed successfully!"
