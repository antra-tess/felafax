#!/bin/bash
set -e

CONDA_PATH=/home/antra_tesserae_cc/miniforge3/bin/conda
PIP_PATH=/home/antra_tesserae_cc/miniforge3/envs/felafax_env/bin/pip
ENV_NAME=felafax_env
REPO_DIR="$HOME/felafax_repo"

# Activate conda environment
if ! source /home/antra_tesserae_cc/miniforge3/bin/activate $ENV_NAME; then
    echo "Failed to activate conda environment $ENV_NAME"
    exit 1
fi

echo "Conda environment activated successfully"
echo "Installing requirements..."
cd "$REPO_DIR"

# Set TPU environment variable if not already set
if ! grep -q "export PJRT_DEVICE=TPU" ~/.bashrc; then
    echo 'export PJRT_DEVICE=TPU' >> ~/.bashrc
    echo "Added PJRT_DEVICE environment variable"
fi

# Install specific versions of dependencies
$PIP_PATH install --no-cache-dir transformers==4.43.3
$PIP_PATH install --no-cache-dir datasets==2.18.0
$PIP_PATH install --no-cache-dir flax
$PIP_PATH install --no-cache-dir einops
$PIP_PATH install --no-cache-dir optax
$PIP_PATH install --no-cache-dir chex
$PIP_PATH install --no-cache-dir absl-py

# Install JAX with TPU support
$PIP_PATH install --upgrade jax
$PIP_PATH install "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html

# Install package in editable mode with dependencies
$PIP_PATH install -e .

echo "Environment setup complete with TPU-specific configurations"
