#!/bin/bash

show_help() {
  echo "Usage: $0 [-f|--force] <script> [arguments]"
  echo
  echo "Arguments:"
  echo "  <script>      The script to run inside the Docker container."
  echo "  [arguments]   Arguments to pass to the script."
  echo
  echo "Options:"
  echo "  -f, --force   Force rebuild the Docker image."
  echo
  echo "Example:"
  echo "  $0 python /mnt/slippi-ai/scripts/run_evaluator.py"
}

FORCE_REBUILD=false

# Parse options and arguments
while [[ "$1" == "-"* ]]; do
  case "$1" in
    -f|--force)
      FORCE_REBUILD=true
      shift
      ;;
    -*)
      show_help
      exit 1
      ;;
  esac
done

if [ $# -lt 1 ]; then
  show_help
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$ROOT_DIR" || { echo "Failed to change directory to $ROOT_DIR"; exit 1; }

HASH_FILE="$SCRIPT_DIR/.file_hashes.txt"

get_check_files() {
  local ignore_file=".dockerignore"
  local check_files=()

  if [ -f $ignore_file ]; then
    while IFS= read -r line; do
      [[ -z "$line" || "$line" =~ ^# ]] && continue
      if [[ "$line" =~ ^! ]]; then
        check_files+=("${line:1}")
      fi
    done < $ignore_file
  fi

  echo "${check_files[@]}"
}

calculate_hashes() {
  local files=("$@")
  for file in "${files[@]}"; do
    if [ -e "$file" ]; then
      find "$file" -type f -exec sha256sum {} + | sort -k 2 | sha256sum
    fi
  done
}

CHECK_FILES=($(get_check_files))

if [ -f $HASH_FILE ]; then
  OLD_HASHES=$(cat $HASH_FILE)
else
  OLD_HASHES=""
fi

CURRENT_HASHES=$(calculate_hashes "${CHECK_FILES[@]}")

if [ "$FORCE_REBUILD" = true ] || ! docker image inspect "slippi-ai:latest" > /dev/null 2>&1; then
  echo "Rebuilding the Docker image..."
  ./build_script.sh
  if [ $? -ne 0 ]; then
    echo "Build failed, exiting."
    exit 1
  fi
fi

if [ "$OLD_HASHES" != "$CURRENT_HASHES" ]; then
  echo "$CURRENT_HASHES" > $HASH_FILE
  SCRIPT_COMMAND="pip install . && $@"
else
  SCRIPT_COMMAND="$@"
fi

docker run --gpus all --rm --name slippi-ai -v "$ROOT_DIR":/mnt/slippi-ai slippi-ai:latest bash -c "$SCRIPT_COMMAND"
if [ $? -ne 0 ]; then
  echo "Failed to run the slippi-ai container"
  exit 1
fi

echo "slippi-ai container ran successfully and has been removed."
