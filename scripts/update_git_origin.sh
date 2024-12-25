#!/bin/bash
set -e

POD_NAME="finetune-70b"
ZONE="us-central2-b"
NUM_WORKERS=8
NEW_ORIGIN="git@github.com:antra-tess/felafax.git"

echo "Updating git origin on all workers..."

update_origin() {
    local worker=$1
    echo "Updating origin on worker $worker..."
    gcloud compute tpus tpu-vm ssh $POD_NAME --zone=$ZONE --worker=$worker --command="
        cd \$HOME/felafax_repo && \
        git remote set-url origin $NEW_ORIGIN && \
        git fetch origin && \
        git reset --hard origin/main
    " &
}

# Start updates on all workers in parallel
for worker in $(seq 0 $((NUM_WORKERS-1))); do
    update_origin $worker
done

# Wait for all updates to complete
wait

echo "Git origin updated on all workers!"
