#!/bin/bash

HEADLESS_DOCKERFILE="./docker_slippi/Dockerfile_headless"
FINAL_DOCKERFILE="./Dockerfile"

echo "Building slippi-emulator:headless image..."
docker build -t slippi-emulator:headless -f $HEADLESS_DOCKERFILE ./docker_slippi
if [ $? -ne 0 ]; then
    echo "Failed to build slippi-emulator:headless image"
    exit 1
fi

echo "slippi-emulator:headless image built successfully."

echo "Building final Docker image..."
docker build --no-cache -t slippi-ai:latest -f $FINAL_DOCKERFILE .
if [ $? -ne 0 ]; then
    echo "Failed to build slippi-ai Docker image"
    exit 1
fi

echo "Final Docker image built successfully."
