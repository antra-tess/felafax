#!/bin/bash
set -e

POD_NAME="finetune-70b"
ZONE="us-central2-b"
NUM_WORKERS=8

echo "Creating conda environments on all workers..."

for worker in $(seq 0 $((NUM_WORKERS-1))); do
    echo "Setting up environment on worker $worker..."
    
    # Create scripts directory
    gcloud compute tpus tpu-vm ssh $POD_NAME --zone=$ZONE --worker=$worker --command="mkdir -p /tmp/felafax_scripts"
    
    # Copy environment script
    gcloud compute tpus tpu-vm scp scripts/create_env.sh $POD_NAME:/tmp/felafax_scripts/ --zone=$ZONE --worker=$worker
    
    # Make executable and run
    gcloud compute tpus tpu-vm ssh $POD_NAME --zone=$ZONE --worker=$worker --command="chmod +x /tmp/felafax_scripts/create_env.sh && /tmp/felafax_scripts/create_env.sh"
done

echo "Conda environments created on all workers"
