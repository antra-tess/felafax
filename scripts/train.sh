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

# Create directories on shared disk (only need to do this once since it's shared)
echo "Creating shared directories..."
mkdir -p $TRAINER_DIR $CHECKPOINT_DIR $EXPORT_DIR

# Create and copy the training script
cat > /tmp/run_training.py << 'EOL'
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import jax
from transformers import AutoTokenizer
from felafax.trainer_engine import checkpoint, trainer, utils
from felafax.trainer_engine.data import data

# Configuration
MODEL_NAME = "/mnt/disk2/llama-3.1-8b"
TRAINER_DIR = "/mnt/disk2/felafax_runs"
CHECKPOINT_DIR = os.path.join(TRAINER_DIR, "checkpoints")
EXPORT_DIR = os.path.join(TRAINER_DIR, "finetuned_export")

# Dataset configuration
dataset_config = data.DatasetConfig(
    data_source="yahma/alpaca-cleaned",
    max_seq_length=512,
    batch_size=32,
    num_workers=4,
    mask_prompt=False,
    train_test_split=0.15,
    ignore_index=-100,
    pad_id=0,
    seed=42
)

# Trainer configuration
trainer_config = trainer.TrainerConfig(
    model_name=MODEL_NAME,
    param_dtype="bfloat16",
    compute_dtype="bfloat16",
    num_epochs=1,
    num_steps=100,  # Initial test with 100 steps
    use_lora=True,
    lora_rank=8,
    learning_rate=1e-4,
    log_interval=10,
    num_tpus=jax.device_count(),
    eval_interval=50,
    eval_steps=5,
    base_dir=TRAINER_DIR
)

# Checkpointer configuration
checkpointer_config = checkpoint.CheckpointerConfig(
    checkpoint_dir=CHECKPOINT_DIR,
    max_to_keep=2,
    save_interval_steps=50,
    create=True,
    enable_async_checkpointing=True
)
checkpointer = checkpoint.Checkpointer(config=checkpointer_config)

# Load tokenizer and prepare datasets
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
train_data, val_data = data.load_data(config=dataset_config)

train_dataset = data.SFTDataset(
    config=dataset_config,
    data=train_data,
    tokenizer=tokenizer
)
val_dataset = data.SFTDataset(
    config=dataset_config,
    data=val_data,
    tokenizer=tokenizer
)

train_dataloader = data.create_dataloader(
    config=dataset_config,
    dataset=train_dataset,
    shuffle=True
)
val_dataloader = data.create_dataloader(
    config=dataset_config,
    dataset=val_dataset,
    shuffle=False
)

# Initialize trainer and start training
trainer = trainer.Trainer(
    trainer_config=trainer_config,
    train_dataloader=train_dataloader,
    val_dataloader=val_dataloader,
    checkpointer=checkpointer
)

print("Starting training...")
trainer.train()

print("Exporting model...")
trainer.export(export_dir=EXPORT_DIR)
EOL

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
