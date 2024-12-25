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

# Create directories on shared disk
for worker in $(seq 0 $((NUM_WORKERS-1))); do
    echo "Setting up directories on worker $worker..."
    gcloud compute tpus tpu-vm ssh $POD_NAME --zone=$ZONE --worker=$worker --command="
        mkdir -p $TRAINER_DIR $CHECKPOINT_DIR $EXPORT_DIR
    "
done

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

# Copy the training script to all workers
for worker in $(seq 0 $((NUM_WORKERS-1))); do
    echo "Copying training script to worker $worker..."
    gcloud compute tpus tpu-vm scp /tmp/run_training.py $POD_NAME:/tmp/run_training.py --zone=$ZONE --worker=$worker
done

# Run training on worker 0 (main worker)
echo "Starting training on worker 0..."
gcloud compute tpus tpu-vm ssh $POD_NAME --zone=$ZONE --worker=0 --command="
    source /home/antra_tesserae_cc/miniforge3/bin/activate felafax_env && \
    cd \$HOME/felafax_repo && \
    python /tmp/run_training.py
"

echo "Training complete!"
