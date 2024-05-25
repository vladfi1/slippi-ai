#!/bin/bash

# Function to display help message
show_help() {
    echo "Usage: $0 <script> [arguments]"
    echo
    echo "Arguments:"
    echo "  <script>      The script to run inside the Docker container."
    echo "  [arguments]   Arguments to pass to the script."
    echo
    echo "Example:"
    echo "  $0 python /mnt/slippi-ai/scripts/run_evaluator.py"
}

# Check if at least one argument is provided
if [ $# -lt 1 ]; then
    show_help
    exit 1
fi

# Capture the command and arguments
SCRIPT_COMMAND="$@"

# Define the hash file
HASH_FILE=".file_hashes.txt"

# Function to parse .dockerignore and derive the files to check for changes
get_check_files() {
    local ignore_file=".dockerignore"
    local check_files=()

    if [ -f $ignore_file ]; then
        while IFS= read -r line; do
            # Skip empty lines and comments
            [[ -z "$line" || "$line" =~ ^# ]] && continue
            # Add lines starting with "!" to the check_files array
            if [[ "$line" =~ ^! ]]; then
                check_files+=("${line:1}")
            fi
        done < $ignore_file
    fi

    echo "${check_files[@]}"
}

# Function to calculate the hash of the specified files
calculate_hashes() {
    local files=("$@")
    for file in "${files[@]}"; do
        if [ -e "$file" ]; then
            find "$file" -type f -exec sha256sum {} + | sort -k 2 | sha256sum
        fi
    done
}

# Get the list of files to check from .dockerignore
CHECK_FILES=($(get_check_files))

# Check if the hash file exists
if [ -f $HASH_FILE ]; then
    OLD_HASHES=$(cat $HASH_FILE)
else
    OLD_HASHES=""
fi

# Calculate the current hashes
CURRENT_HASHES=$(calculate_hashes "${CHECK_FILES[@]}")

# Compare old and new hashes
if [ "$OLD_HASHES" != "$CURRENT_HASHES" ]; then
    echo "Changes detected in monitored files. Rebuilding the Docker image..."
    echo "$CURRENT_HASHES" > $HASH_FILE
    ./build_script.sh
    if [ $? -ne 0 ]; then
        echo "Build failed, exiting."
        exit 1
    fi
else
    echo "No changes detected in monitored files. Skipping the build."
fi

# Run the Docker container with GPU support and mount the current working directory
docker run --gpus all --rm --name slippi-ai -v $(pwd):/mnt/slippi-ai slippi-ai:latest $SCRIPT_COMMAND
if [ $? -ne 0 ]; then
    echo "Failed to run the slippi-ai container"
    exit 1
fi

echo "slippi-ai container ran successfully and has been removed."
