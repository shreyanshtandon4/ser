#!/bin/bash

# Source directory containing multiple subfolders
SOURCE_DIR="train_data"

# Destination directory where all files will be moved
DEST_DIR="training"

# Create destination directory if it doesn't exist
mkdir -p "$DEST_DIR"

# Find and move all files from subfolders to the destination directory
find "$SOURCE_DIR" -type f -exec mv {} "$DEST_DIR" \;

echo "All files moved successfully to $DEST_DIR"
