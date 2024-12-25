#!/bin/bash
set -e

CONDA_PATH=/opt/conda/bin/conda
ENV_NAME=felafax_env
PYTHON_VERSION=3.10

echo "Creating conda environment $ENV_NAME..."
$CONDA_PATH create -n $ENV_NAME python=$PYTHON_VERSION -y

echo "Environment created successfully"
