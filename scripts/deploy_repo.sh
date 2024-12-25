#!/bin/bash
set -e

POD_NAME="finetune-70b"
ZONE="us-central2-b"
NUM_WORKERS=8

echo "Cloning repository on all workers..."

for worker in $(seq 0 $((NUM_WORKERS-1))); do
    echo "Cloning repo on worker $worker..."
    
    # Copy clone script
    gcloud compute tpus tpu-vm scp scripts/clone_repo.sh $POD_NAME:/tmp/felafax_scripts/ --zone=$ZONE --worker=$worker
    
    # Make executable and run
    gcloud compute tpus tpu-vm ssh $POD_NAME --zone=$ZONE --worker=$worker --command="chmod +x /tmp/felafax_scripts/clone_repo.sh && /tmp/felafax_scripts/clone_repo.sh"
done

echo "Repository cloned on all workers"
