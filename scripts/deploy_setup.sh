#!/bin/bash
set -e

POD_NAME="finetune-70b"
ZONE="us-central2-b"
NUM_WORKERS=8

echo "Setting up environments on all workers..."

for worker in $(seq 0 $((NUM_WORKERS-1))); do
    echo "Running setup on worker $worker..."
    
    # Copy setup script
    gcloud compute tpus tpu-vm scp scripts/setup_environment.sh $POD_NAME:/tmp/felafax_scripts/ --zone=$ZONE --worker=$worker
    
    # Make executable and run
    gcloud compute tpus tpu-vm ssh $POD_NAME --zone=$ZONE --worker=$worker --command="chmod +x /tmp/felafax_scripts/setup_environment.sh && /tmp/felafax_scripts/setup_environment.sh"
done

echo "Environment setup complete on all workers"
