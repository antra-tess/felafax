#!/bin/bash
set -e

REPO_URL="git@github.com:antra-tess/felafax.git"
TARGET_DIR="$HOME/felafax_repo"

echo "Cloning/updating Felafax repository..."
if [ -d "$TARGET_DIR" ]; then
    echo "Directory exists, fetching and resetting to latest changes..."
    cd "$TARGET_DIR"
    git fetch origin
    git reset --hard origin/main
else
    echo "Cloning fresh repository..."
    git clone "$REPO_URL" "$TARGET_DIR"
fi

echo "Repository setup complete"
