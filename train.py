import argparse
import torch
from itertools import islice
import os

from dataset import get_dataset # Added import
from moderngpt2 import (
    ModernGPT2Config,
    ModernGPT2LMHeadModel,
)
from transformers import (
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
# Launch with: accelerate launch --config_file accelerate_config.yaml train.py --ds_config ds_config.json ...
# Or: deepspeed train.py --deepspeed --deepspeed_config ds_config.json --ds_config ds_config.json ...

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
    parser.add_argument("--report_to", type=str, default="none", help="Report metrics to (e.g., 'wandb', 'tensorboard', 'none'). Default: 'none'.")
    parser.add_argument("--wandb_project", type=str, default="moderngpt2", help="W&B project name. Default: 'moderngpt2'.")
    parser.add_argument("--ds_config", type=str, default=None, help="Path to DeepSpeed config JSON for Hugging Face Trainer.")
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
    parser.add_argument(
        "--metadata_file",
        type=str,
        default=None,
        help="Path to YAML metadata file containing train/eval parquet file lists. Takes precedence over pre_tokenized_dataset_path.",
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training. Passed by launch script.")
    parser.add_argument("--fp16", action="store_true", help="Enable mixed precision training (fp16).")
    parser.add_argument("--bf16", action="store_true", help="Enable mixed precision training with bfloat16 (requires Ampere or newer GPU). Mutually exclusive with --fp16.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--tf32", type=bool, default=True, help="Enable TF32 on Ampere GPUs for faster training.")
    parser.add_argument("--gradient_checkpointing", action="store_true", help="Enable gradient checkpointing to save memory.")
    parser.add_argument("--torch_compile", action="store_true", help="Enable torch.compile for model optimization.")
    parser.add_argument("--torch_compile_backend", type=str, default="inductor", help="Backend for torch.compile.")
    parser.add_argument("--torch_compile_mode", type=str, default="default", choices=["default", "reduce-overhead", "max-autotune"], help="Compilation mode for torch.compile.")
    parser.add_argument("--dataloader_num_workers", type=int, default=4, help="Number of data loading workers.")
    parser.add_argument("--dataloader_prefetch_factor", type=int, default=2, help="Number of batches to prefetch per worker.")
    parser.add_argument("--dataloader_pin_memory", action="store_true", help="Pin memory for faster GPU transfer.")
    parser.add_argument("--warmup_steps", type=int, default=None, help="Number of warmup steps. Overrides warmup_ratio if set.")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Max gradient norm for clipping.")

    args = parser.parse_args()

    if args.fp16 and args.bf16:
        raise ValueError("Cannot use both --fp16 and --bf16 flags simultaneously. Please choose one.")

    # Setup logging
    logging.set_verbosity_info()
    
    # Enable TF32 for Ampere GPUs
    if args.tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        logger.info("TF32 enabled for matrix multiplications")
    
    # Set NCCL environment variables for better multi-GPU performance
    if torch.cuda.device_count() > 1:
        os.environ.setdefault('NCCL_ASYNC_ERROR_HANDLING', '1')
        os.environ.setdefault('NCCL_TIMEOUT', '1800')  # 30 minutes

    # --- REMOVED OLD TOKENIZER LOADING ---
    # --- REMOVED OLD DATASET HANDLING ---

    # --- ADD NEW DATASET AND TOKENIZER LOADING ---
    if args.metadata_file:
        logger.info(
            f"Loading dataset from metadata file: {args.metadata_file}, Tokenizer: {args.tokenizer_path}"
        )
    else:
        logger.info(
            f"Loading dataset. Tokenizer: {args.tokenizer_path}, Block size: {args.block_size}, "
            f"Pre-tokenized path: {args.pre_tokenized_dataset_path if args.pre_tokenized_dataset_path else 'Not provided (using C4 streaming)'}"
        )
    
    streaming_dataset = not (args.pre_tokenized_dataset_path or args.metadata_file)  # True if using C4, False if using pre-tokenized or metadata
    train_dataset, eval_dataset, tokenizer = get_dataset(
        tokenizer_path=args.tokenizer_path,
        block_size=args.block_size,
        streaming=streaming_dataset,
        pre_tokenized_path=args.pre_tokenized_dataset_path,
        streaming_eval_samples=args.streaming_eval_samples,
        metadata_file=args.metadata_file
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
    
    # Enable gradient checkpointing if requested
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        logger.info("Gradient checkpointing enabled")
    
    # Apply torch.compile if requested and available
    if args.torch_compile and hasattr(torch, 'compile'):
        logger.info(f"Compiling model with torch.compile (backend={args.torch_compile_backend}, mode={args.torch_compile_mode})")
        model = torch.compile(
            model, 
            backend=args.torch_compile_backend,
            mode=args.torch_compile_mode
        )

    # Set up W&B if enabled
    if args.report_to == "wandb":
        import wandb
        wandb.init(project=args.wandb_project, reinit=True)
        logger.info(f"W&B logging enabled with project: {args.wandb_project}")
    
    # TrainingArguments
    logger.info("Setting up TrainingArguments...")
    training_args_dict = {
        "output_dir": args.output_dir,
        "num_train_epochs": args.num_train_epochs,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "logging_steps": args.logging_steps,
        "save_steps": args.save_steps,
        "eval_steps": args.eval_steps,
        "save_total_limit": args.save_total_limit,
        "report_to": args.report_to if args.report_to != "none" else None,  # Handle "none" case
        "deepspeed": args.ds_config,
        "fp16": args.fp16,
        "bf16": args.bf16,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "dataloader_num_workers": args.dataloader_num_workers,
        "dataloader_prefetch_factor": args.dataloader_prefetch_factor,
        "dataloader_pin_memory": args.dataloader_pin_memory,
        "max_grad_norm": args.max_grad_norm,
        "tf32": args.tf32,
        "gradient_checkpointing": args.gradient_checkpointing,
        # Optimizer settings for better performance
        "optim": "adamw_torch_fused" if torch.cuda.is_available() else "adamw_torch",
        "adam_beta1": 0.9,
        "adam_beta2": 0.95,
        "adam_epsilon": 1e-8,
    }
    
    # Handle warmup_steps vs warmup_ratio
    if args.warmup_steps is not None:
        training_args_dict["warmup_steps"] = args.warmup_steps
    else:
        training_args_dict["warmup_ratio"] = args.warmup_ratio
    
    training_args = TrainingArguments(**training_args_dict)

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
