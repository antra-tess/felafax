# Setup
```
```python
!pip install git+https://github.com/felafax/felafax.git -q
!pip uninstall -y tensorflow && pip install tensorflow-cpu -q
```
```python
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
```
```python
MODEL_NAME = "meta-llama/Llama-3.2-1B"
HF_TOKEN = input("Please enter your HuggingFace token: ")
TRAINER_DIR = "/"
TEST_MODE = False


CHECKPOINT_DIR = os.path.join(TRAINER_DIR, "checkpoints")
EXPORT_DIR = os.path.join(TRAINER_DIR, "finetuned_export")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(EXPORT_DIR, exist_ok=True)
```
```python
from felafax.trainer_engine import setup
setup.setup_environment(base_dir=TRAINER_DIR)

import jax
from transformers import AutoTokenizer
from felafax.trainer_engine import checkpoint, trainer, utils
from felafax.trainer_engine.data import data
```
```markdown
# Step 0: Configure different parts of training pipeline
```
```python
dataset_config = data.DatasetConfig(
    data_source="yahma/alpaca-cleaned",
    max_seq_length=32,
    batch_size=8,
    num_workers=4,
    mask_prompt=False,
    train_test_split=0.15,

    ignore_index=-100,
    pad_id=0,
    seed=42,

    # Setting max_examples limits the number of examples in the dataset.
    # This is useful for testing the pipeline without running the entire dataset.
    max_examples=100 if TEST_MODE else None,
)
```
```python
trainer_config = trainer.TrainerConfig(
    model_name=MODEL_NAME,
    param_dtype="bfloat16",
    compute_dtype="bfloat16",

    # Training configuration
    num_epochs=1,
    num_steps=50,
    use_lora=True,
    lora_rank=16,
    learning_rate=1e-3,
    log_interval=1,

    num_tpus=jax.device_count(),

    # Eval configuration
    eval_interval=50,
    eval_steps=5,

    # Additional info required by trainer
    base_dir=TRAINER_DIR,
    hf_token=HF_TOKEN,
)
```
```python
checkpointer_config = checkpoint.CheckpointerConfig(
    checkpoint_dir=CHECKPOINT_DIR,
    max_to_keep=2,
    save_interval_steps=50,
    erase_existing_checkpoints=True,
)
checkpointer = checkpoint.Checkpointer(config=checkpointer_config)
```
```markdown
# Step 1: Downloading dataset...
```
```markdown
For this colab, we're utilizing the refined **Alpaca dataset**, curated by yahma. This dataset is a carefully filtered selection of 52,000 entries from the original Alpaca collection. Feel free to substitute this section with your own data preparation code if you prefer.

It's crucial to include the EOS_TOKEN (End of Sequence Token) in your tokenized output. Failing to do so may result in endless generation loops.
```
```python
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN)

# Download and load the data files
train_data, val_data = data.load_data(config=dataset_config)

# Create datasets for SFT (supervised fine-tuning)
train_dataset = data.SFTDataset(
    config=dataset_config,
    data=train_data,
    tokenizer=tokenizer,
)
val_dataset = data.SFTDataset(
    config=dataset_config,
    data=val_data,
    tokenizer=tokenizer,
)

# Create dataloaders
train_dataloader = data.create_dataloader(
    config=dataset_config,
    dataset=train_dataset,
    shuffle=True,
)
val_dataloader = data.create_dataloader(
    config=dataset_config,
    dataset=val_dataset,
    shuffle=False,
)
```
```markdown
# Step 2: Create Trainer and load the model
```
```python
trainer = trainer.Trainer(
    trainer_config=trainer_config,
    train_dataloader=train_dataloader,
    val_dataloader=val_dataloader,
    checkpointer=checkpointer,
)
```
```python
trainer.train()
```
```markdown
# Step 3: Export fine-tuned model
```
```python
trainer.export(export_dir=EXPORT_DIR)
```
```python
utils.upload_dir_to_hf(
    dir_path=EXPORT_DIR,
    repo_name="felarof01/test-llama3-alpaca-from-colab",
    token=HF_TOKEN,
)
```
