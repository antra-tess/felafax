#!/bin/bash
set -e

POD_NAME="finetune-70b"
ZONE="us-central2-b"
NUM_WORKERS=8

echo "Deploying to TPU pod $POD_NAME..."

# Function to run command on a worker
run_on_worker() {
    local worker=$1
    local cmd=$2
    echo "Running on worker-$worker: $cmd"
    gcloud compute tpus tpu-vm ssh $POD_NAME --zone=$ZONE --worker=$worker --command="$cmd"
}

# Function to copy file to a worker
copy_to_worker() {
    local worker=$1
    local src=$2
    local dst=$3
    echo "Copying to worker-$worker: $src -> $dst"
    gcloud compute tpus tpu-vm scp $src $POD_NAME:$dst --zone=$ZONE --worker=$worker
}

# Deploy to each worker
for worker in $(seq 0 $((NUM_WORKERS-1))); do
    echo "Setting up worker $worker..."
    
    # Create scripts directory
    run_on_worker $worker "mkdir -p /tmp/felafax_scripts"
    
    # Copy deployment scripts
    copy_to_worker $worker "scripts/create_env.sh" "/tmp/felafax_scripts/"
    copy_to_worker $worker "scripts/clone_repo.sh" "/tmp/felafax_scripts/"
    copy_to_worker $worker "scripts/setup_environment.sh" "/tmp/felafax_scripts/"
    
    # Make scripts executable
    run_on_worker $worker "chmod +x /tmp/felafax_scripts/*.sh"
    
    # Run scripts in sequence
    echo "Creating conda environment..."
    run_on_worker $worker "/tmp/felafax_scripts/create_env.sh"
    
    echo "Cloning repository..."
    run_on_worker $worker "/tmp/felafax_scripts/clone_repo.sh"
    
    echo "Setting up environment..."
    run_on_worker $worker "/tmp/felafax_scripts/setup_environment.sh"
done

echo "Deployment complete!"
