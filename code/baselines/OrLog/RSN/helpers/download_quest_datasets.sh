#!/bin/bash

# Script to download QUEST datasets and document corpus

# Define directories
BASE_DIR=$(dirname "$0")/..
DATASET_DIR="$BASE_DIR/datasets/QUEST"

# Create dataset directory if it doesn't exist
mkdir -p "$DATASET_DIR"

# Define the files and their URLs
declare -A files
files=(
    ["train.jsonl"]="https://storage.googleapis.com/gresearch/quest/train.jsonl"
    ["train_aug.jsonl"]="https://storage.googleapis.com/gresearch/quest/train_aug.jsonl"
    ["val.jsonl"]="https://storage.googleapis.com/gresearch/quest/val.jsonl"
    ["test.jsonl"]="https://storage.googleapis.com/gresearch/quest/test.jsonl"
    ["documents.zip"]="https://storage.googleapis.com/gresearch/quest/documents.zip"
)

# Download files if they don't exist
echo "Checking for QUEST datasets..."
for file in "${!files[@]}"; do
    if [ -f "$DATASET_DIR/$file" ]; then
        echo "$file already exists, skipping download."
    else
        echo "Downloading $file..."
        wget -O "$DATASET_DIR/$file" "${files[$file]}"
    fi
done

# Extract documents.zip if necessary
if [ -f "$DATASET_DIR/documents.zip" ]; then
    if [ ! -d "$DATASET_DIR/documents" ]; then
        echo "Extracting document corpus..."
        unzip -o "$DATASET_DIR/documents.zip" -d "$DATASET_DIR"
        rm "$DATASET_DIR/documents.zip"
    else
        echo "Document corpus already extracted, skipping."
    fi
fi

echo "QUEST datasets and document corpus saved successfully in $DATASET_DIR."