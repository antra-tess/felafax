from transformers import AutoTokenizer
from felafax.trainer_engine.trainer import Trainer, TrainerConfig
from felafax.trainer_engine.setup import setup_environment
from felafax.trainer_engine.checkpoint import Checkpointer, CheckpointerConfig
from .dataset import AlpacaDataset, AlpacaDatasetConfig
from felafax.trainer_engine import utils

########################################################
# Configure the dataset pipeline
########################################################
tokenizer = AutoTokenizer.from_pretrained(
    "meta-llama/Llama-3.2-1B", token="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"
)
dataset_config = AlpacaDatasetConfig(
    data_source="yahma/alpaca-cleaned",
    max_seq_length=32,
    batch_size=8,
    num_workers=4,
    mask_prompt=False,
    train_test_split=0.15,
    max_examples=100,  # Set to an integer to limit examples
    seed=42,
)
alpaca_dataset = AlpacaDataset(config=dataset_config)
alpaca_dataset.setup(tokenizer=tokenizer)

train_dataloader = alpaca_dataset.train_dataloader()
val_dataloader = alpaca_dataset.val_dataloader()


########################################################
# Configure the trainer pipeline
########################################################
trainer_config = TrainerConfig(
    model_name="meta-llama/Llama-3.2-1B",
    hf_token="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",
    num_steps=5,
    num_tpus=1,
    base_dir="/Users/felarof99/Workspaces/GITHUB/building/",
)

# Set up the training environment using trainer_config
setup_environment(trainer_config)

# Configure the checkpointer
checkpointer_config = CheckpointerConfig(
    checkpoint_dir=f"{trainer_config.base_dir}/checkpoints/",
)
checkpointer = Checkpointer(config=checkpointer_config)

# Put everything together and initialize the trainer
trainer = Trainer(
    trainer_config=trainer_config,
    train_dataloader=train_dataloader,
    val_dataloader=val_dataloader,
    checkpointer=checkpointer,
)

# Run training
trainer.train()

# Upload exported model to HF
utils.upload_dir_to_hf(
    dir_path=f"{trainer_config.base_dir}/hf_export/",
    repo_name="felarof01/test-llama3-alpaca",
)
