#!/bin/bash
set -e

CONDA_PATH=/home/antra_tesserae_cc/miniforge3/bin/conda
PIP_PATH=/home/antra_tesserae_cc/miniforge3/envs/felafax_env/bin/pip
ENV_NAME=felafax_env
REPO_DIR="$HOME/felafax_repo"

# Activate conda environment
source /home/antra_tesserae_cc/miniforge3/bin/activate $ENV_NAME

echo "Installing requirements..."
cd "$REPO_DIR"

# Install package in editable mode with dependencies
$PIP_PATH install -e .

echo "Environment setup complete"
