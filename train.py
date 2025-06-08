import argparse
import torch
from itertools import islice

from dataset import get_dataset # Added import
from transformers import (
    ModernGPT2Config,
    ModernGPT2LMHeadModel,
    ModernGPT2TokenizerFast,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from transformers.utils import logging # For logger if needed

logger = logging.get_logger(__name__)

# Example ds_config.json for ZeRO-1:
# {
#   "zero_optimization": { "stage": 1 },
#   "optimizer": { "type": "AdamW", "params": { "lr": "auto"}},
#   "fp16": { "enabled": true }
# }
# Launch with: accelerate launch --config_file accelerate_config.yaml train.py --deepspeed_config ds_config.json ...
# Or: deepspeed train.py --deepspeed --deepspeed_config ds_config.json ...

def main():
    parser = argparse.ArgumentParser(description="Train ModernGPT2 Model")
    parser.add_argument(
        "--model_size_name",
        type=str,
        default="small",
        choices=["small", "medium", "large", "xl"],
        help="Size of the ModernGPT2 model to train.",
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default="moderngpt2-tokenizer-sp", # Placeholder, user should provide path to their trained SentencePiece model
        help="Path to the SentencePiece tokenizer model file.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output_dir",
        help="Output directory for model checkpoints and logs.",
    )
    parser.add_argument("--num_train_epochs", type=int, default=1, help="Number of training epochs.")
    parser.add_argument(
        "--per_device_train_batch_size", type=int, default=8, help="Batch size per device during training."
    )
    parser.add_argument(
        "--per_device_eval_batch_size", type=int, default=8, help="Batch size per device during evaluation."
    )
    parser.add_argument("--learning_rate", type=float, default=2.5e-4, help="Learning rate.")
    parser.add_argument("--warmup_ratio", type=float, default=0.05, help="Warmup ratio for learning rate scheduler.")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay.")
    parser.add_argument("--logging_steps", type=int, default=500, help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=5000, help="Save checkpoint every X updates steps.")
    parser.add_argument("--eval_steps", type=int, default=500, help="Evaluate every X updates steps.")
    parser.add_argument("--save_total_limit", type=int, default=10, help="Limit the total number of checkpoints.")
    parser.add_argument("--report_to", type=str, default="wandb", help="Report metrics to (e.g., 'wandb', 'tensorboard').")
    parser.add_argument("--deepspeed_config", type=str, default=None, help="Path to DeepSpeed config JSON.")
    parser.add_argument(
        "--streaming_eval_samples", type=int, default=1000, help="Number of samples for streaming evaluation."
    )
    parser.add_argument("--block_size", type=int, default=1024, help="Block size for tokenization and chunking (used if not loading pre-tokenized data).")
    parser.add_argument(
        "--pre_tokenized_dataset_path",
        type=str,
        default=None,
        help="Path to a directory containing pre-tokenized Parquet files. If provided, C4 streaming is skipped.",
    )

    args = parser.parse_args()

    # Setup logging
    logging.set_verbosity_info()

    # --- REMOVED OLD TOKENIZER LOADING ---
    # --- REMOVED OLD DATASET HANDLING ---

    # --- ADD NEW DATASET AND TOKENIZER LOADING ---
    logger.info(
        f"Loading dataset. Tokenizer: {args.tokenizer_path}, Block size: {args.block_size}, "
        f"Pre-tokenized path: {args.pre_tokenized_dataset_path if args.pre_tokenized_dataset_path else 'Not provided (using C4 streaming)'}"
    )
    train_dataset, eval_dataset, tokenizer = get_dataset(
        tokenizer_path=args.tokenizer_path,
        block_size=args.block_size, # Still needed for C4 streaming, or if model config depends on it
        streaming=True, # This 'streaming' arg in get_dataset applies to C4 or Parquet if supported
        pre_tokenized_path=args.pre_tokenized_dataset_path, # Pass the new argument here
        streaming_eval_samples=args.streaming_eval_samples
    )
    # --- END ADDITION ---

    # Data Collator (uses the new tokenizer)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Configuration and Model (uses the new tokenizer for pad_token_id and vocab_size)
    logger.info(f"Initializing model with size: {args.model_size_name}")
    config = ModernGPT2Config(
        model_size_name=args.model_size_name,
        pad_token_id=tokenizer.pad_token_id, # Use tokenizer from get_dataset
    )
    if config.vocab_size != tokenizer.vocab_size:
        logger.warning(
            f"Config vocab_size ({config.vocab_size}) does not match tokenizer vocab_size ({tokenizer.vocab_size}). "
            f"Setting config.vocab_size to {tokenizer.vocab_size}."
        )
        config.vocab_size = tokenizer.vocab_size

    model = ModernGPT2LMHeadModel(config)

    # TrainingArguments
    logger.info("Setting up TrainingArguments...")
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        evaluation_strategy="steps", # Evaluate at eval_steps
        eval_steps=args.eval_steps,
        save_total_limit=args.save_total_limit,
        report_to=args.report_to if args.report_to != "none" else None, # Handle "none" case
        deepspeed=args.deepspeed_config,
        # Other arguments like adam_beta1, adam_beta2, lr_scheduler_type can be added if needed
        # For streaming, max_steps might be more appropriate than num_train_epochs depending on dataset size.
        # If the dataset is truly infinite or very large, set max_steps.
        # For this example, num_train_epochs will cause iteration over the 'iterable' train_dataset.
    )

    # Trainer
    logger.info("Initializing Trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    # Train
    logger.info("Starting training...")
    trainer.train()

    logger.info("Training finished.")
    # Optionally, save the final model
    # trainer.save_model(os.path.join(args.output_dir, "final_model"))


if __name__ == "__main__":
    main()
