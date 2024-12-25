#!/bin/bash
set -e

POD_NAME="finetune-70b"
ZONE="us-central2-b"
NUM_WORKERS=8

# Configuration
MODEL_NAME="/mnt/disk2/llama-3.1-8b"
TRAINER_DIR="/mnt/disk2/felafax_runs"
CHECKPOINT_DIR="$TRAINER_DIR/checkpoints"
EXPORT_DIR="$TRAINER_DIR/finetuned_export"

# Create directories on shared disk using the first worker
echo "Creating shared directories using worker 0..."
gcloud compute tpus tpu-vm ssh $POD_NAME --zone=$ZONE --worker=0 --command="
    sudo mkdir -p $TRAINER_DIR $CHECKPOINT_DIR $EXPORT_DIR && \
    sudo chown -R antra_tesserae_cc:antra_tesserae_cc $TRAINER_DIR && \
    sudo chmod -R 775 $TRAINER_DIR
"

# Copy the training script to workers

# Function to run commands on a worker
run_on_worker() {
    local worker=$1
    echo "Setting up worker $worker..."
    # Copy training script
    gcloud compute tpus tpu-vm scp /tmp/run_training.py $POD_NAME:/tmp/run_training.py --zone=$ZONE --worker=$worker &
}

# Start setup on all workers in parallel
for worker in $(seq 0 $((NUM_WORKERS-1))); do
    run_on_worker $worker
done

# Wait for all background processes to complete
wait

# Function to start training on a worker
start_training() {
    local worker=$1
    echo "Starting training on worker $worker..."
    gcloud compute tpus tpu-vm ssh $POD_NAME --zone=$ZONE --worker=$worker --command="
        source /home/antra_tesserae_cc/miniforge3/bin/activate felafax_env && \
        cd \$HOME/felafax_repo && \
        python /tmp/run_training.py
    " &
}

# Start training on all workers in parallel
echo "Starting training on all workers..."
for worker in $(seq 0 $((NUM_WORKERS-1))); do
    start_training $worker
done

# Wait for all training processes to complete
wait

echo "Training complete on all workers!"
