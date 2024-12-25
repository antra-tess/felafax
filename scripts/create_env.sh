#!/bin/bash
set -e

CONDA_PATH=/home/antra_tesserae_cc/miniforge3/bin/conda
ENV_NAME=felafax_env

echo "Creating conda environment $ENV_NAME..."
$CONDA_PATH create -n $ENV_NAME python=3.10 -y -c conda-forge

echo "Environment created successfully"
