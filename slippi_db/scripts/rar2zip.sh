#!/bin/bash

# rar_to_zip.sh
# Usage: ./rar_to_zip.sh yourfile.rar

set -e  # Exit on error

# Check for input
if [ -z "$1" ]; then
  echo "Usage: $0 yourfile.rar"
  exit 1
fi

RAR_FILE="$1"
BASENAME=$(basename "$RAR_FILE" .rar)
EXTRACT_DIR="${BASENAME}_extracted"
ZIP_FILE="${BASENAME}.zip"

# Check dependencies
command -v unrar >/dev/null 2>&1 || { echo >&2 "unrar is not installed. Please install it."; exit 1; }
command -v 7z >/dev/null 2>&1 || { echo >&2 "7z is not installed. Please install it."; exit 1; }

# Make sure the file exists
if [ ! -f "$RAR_FILE" ]; then
  echo "File not found: $RAR_FILE"
  exit 1
fi

# Create extraction directory
mkdir -p "$EXTRACT_DIR"

# Extract the .rar file
echo "Extracting $RAR_FILE..."
unrar x -o+ "$RAR_FILE" "$EXTRACT_DIR/"

# Zip the contents
echo "Creating $ZIP_FILE..."
7z a -tzip -mmt=on "$ZIP_FILE" "$EXTRACT_DIR"/*

rm -rf "$EXTRACT_DIR"

echo "Done! Created $ZIP_FILE"
