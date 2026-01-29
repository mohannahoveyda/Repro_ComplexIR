#!/bin/bash

# Script to download, prepare, and decompress the RoMQA dataset

# Define directories
BASE_DIR=$(dirname "$0")/..
DATASET_DIR="$BASE_DIR/datasets/RoMQA"

# Create dataset directory if it doesn't exist
mkdir -p "$DATASET_DIR"

# Define the URL and file name
URL="https://s3.us-west-1.wasabisys.com/vzhong-public/RoMQA/romqa_data.zip"
ZIP_FILE="$DATASET_DIR/romqa_data.zip"

# Check if the ZIP file exists
if [ -f "$ZIP_FILE" ]; then
    echo "romqa_data.zip already exists, skipping download."
else
    echo "Downloading romqa_data.zip..."
    wget -O "$ZIP_FILE" "$URL"
fi

# Extract the ZIP file if not already extracted
if [ -d "$DATASET_DIR/data" ]; then
    echo "RoMQA data already extracted, skipping."
else
    echo "Extracting RoMQA data..."
    unzip -o "$ZIP_FILE" -d "$DATASET_DIR"
fi

# Decompress .bz2 files in all subdirectories of `data`
echo "Searching for .bz2 files to decompress..."
BZ2_FILES=$(find "$DATASET_DIR/data" -type f -name "*.bz2")

if [ -z "$BZ2_FILES" ]; then
    echo "No .bz2 files found to decompress."
else
    echo "Decompressing .bz2 files..."
    for bz2_file in $BZ2_FILES; do
        decompressed_file="${bz2_file%.bz2}"  # Remove the .bz2 extension
        if [ -f "$decompressed_file" ]; then
            echo "Decompressed file $decompressed_file already exists. Deleting $bz2_file..."
            rm "$bz2_file"
        else
            echo "Decompressing $bz2_file..."
            bzip2 -d "$bz2_file"
        fi
    done
fi

echo "RoMQA dataset prepared and decompressed successfully in $DATASET_DIR."