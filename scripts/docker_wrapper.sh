#!/bin/bash

show_help() {
  echo "Usage: $0 [--rebuild] [--clear-hash] <script> [arguments]"
  echo
  echo "Arguments:"
  echo "  <script>      The script to run inside the Docker container."
  echo "  [arguments]   Arguments to pass to the script."
  echo
  echo "Options:"
  echo "  --rebuild     Force rebuild the Docker image."
  echo
  echo "Example:"
  echo "  $0 python /mnt/slippi-ai/scripts/run_evaluator.py"
}

FORCE_REBUILD=false
CLEAR_HASH=false

# Parse options and arguments
while [[ "$1" == "--"* ]]; do
  case "$1" in
    --rebuild)
      FORCE_REBUILD=true
      shift
      ;;
    --*)
      show_help
      exit 1
      ;;
  esac
done

if [ $# -lt 1 ] && [ "$FORCE_REBUILD" = false ]; then
  show_help
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$ROOT_DIR" || { echo "Failed to change directory to $ROOT_DIR"; exit 1; }

if [ "$FORCE_REBUILD" = true ] || ! docker image inspect "slippi-ai:latest" > /dev/null 2>&1; then
  echo "Rebuilding the Docker image..."
  ./build_script.sh
  if [ $? -ne 0 ]; then
    echo "Build failed, exiting."
    exit 1
  fi
fi

SCRIPT_COMMAND="pip install . && $@"

docker run --gpus all --rm --name slippi-ai -v "$ROOT_DIR":/mnt/slippi-ai slippi-ai:latest bash -c "$SCRIPT_COMMAND"
if [ $? -ne 0 ]; then
  echo "Failed to run the slippi-ai container"
  exit 1
fi

echo "slippi-ai container ran successfully and has been removed."
