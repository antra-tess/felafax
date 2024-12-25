#!/bin/bash
set -e

REPO_URL="https://github.com/antra-tess/felafax"
TARGET_DIR="/mnt/disk2/felafax_repo"

echo "Cloning Felafax repository..."
if [ -d "$TARGET_DIR" ]; then
    echo "Directory exists, pulling latest changes..."
    cd "$TARGET_DIR"
    git pull
else
    echo "Cloning fresh repository..."
    git clone "$REPO_URL" "$TARGET_DIR"
fi

echo "Repository setup complete"
