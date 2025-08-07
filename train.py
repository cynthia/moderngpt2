import argparse
import torch
from itertools import islice
import os
import sys

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
from accelerate import Accelerator
import torch.distributed as dist

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
    parser.add_argument("--max_steps", type=int, default=None, help="Maximum number of training steps. Overrides num_train_epochs if set.")
    parser.add_argument("--total_train_samples", type=int, default=None, help="Total number of training samples (for calculating steps with streaming datasets).")
    parser.add_argument(
        "--per_device_train_batch_size", type=int, default=8, help="Batch size per device during training."
    )
    parser.add_argument(
        "--per_device_eval_batch_size", type=int, default=8, help="Batch size per device during evaluation."
    )
    parser.add_argument("--learning_rate", type=float, default=2.5e-4, help="Learning rate.")
    parser.add_argument("--warmup_ratio", type=float, default=0.05, help="Warmup ratio for learning rate scheduler.")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay.")
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine", help="Learning rate scheduler type. Default: 'cosine'.")
    parser.add_argument("--logging_steps", type=int, default=500, help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=5000, help="Save checkpoint every X updates steps.")
    parser.add_argument("--eval_steps", type=int, default=500, help="Evaluate every X updates steps.")
    parser.add_argument("--save_total_limit", type=int, default=10, help="Limit the total number of checkpoints.")
    parser.add_argument("--report_to", type=str, default="none", help="Report metrics to (e.g., 'wandb', 'tensorboard', 'none'). Default: 'none'.")
    parser.add_argument("--wandb_project", type=str, default="moderngpt2", help="W&B project name. Default: 'moderngpt2'.")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="W&B run name. If not provided, will auto-generate based on model size.")
    parser.add_argument("--ds_config", "--deepspeed_config", type=str, default=None, help="Path to DeepSpeed config JSON for Hugging Face Trainer.")
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
    parser.add_argument("--dataset_cache_dir", type=str, default=".dataset_cache", help="Directory to cache concatenated datasets.")
    parser.add_argument("--use_dataset_cache", action="store_true", help="Enable dataset caching for faster subsequent runs.")
    parser.add_argument("--clear_dataset_cache", action="store_true", help="Clear the dataset cache before training.")
    parser.add_argument("--streaming", action="store_true", help="Enable streaming mode for pre-tokenized datasets to avoid caching.")
    parser.add_argument("--force_no_streaming", action="store_true", help="Force disable streaming even for large datasets (may cause memory issues).")

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
    # Initialize Accelerator to get process rank info
    accelerator = Accelerator()
    
    # Clear cache if requested (only on rank 0)
    if args.clear_dataset_cache and accelerator.is_main_process:
        from dataset_cache import DatasetCache
        cache = DatasetCache(args.dataset_cache_dir)
        cache.clear_cache()
        logger.info("Dataset cache cleared")
    
    # Synchronize all processes after cache clearing
    if args.clear_dataset_cache:
        accelerator.wait_for_everyone()
    
    if args.metadata_file:
        logger.info(
            f"Loading dataset from metadata file: {args.metadata_file}, Tokenizer: {args.tokenizer_path}"
        )
        if args.use_dataset_cache:
            logger.info(f"Dataset caching enabled, cache directory: {args.dataset_cache_dir}")
    else:
        logger.info(
            f"Loading dataset. Tokenizer: {args.tokenizer_path}, Block size: {args.block_size}, "
            f"Pre-tokenized path: {args.pre_tokenized_dataset_path if args.pre_tokenized_dataset_path else 'Not provided (using C4 streaming)'}"
        )
    
    # Determine streaming mode
    if args.force_no_streaming:
        streaming_dataset = False
        logger.info("Streaming disabled by --force_no_streaming flag")
    elif args.streaming:
        streaming_dataset = True
        logger.info("Streaming enabled by --streaming flag")
    elif args.pre_tokenized_dataset_path or args.metadata_file:
        # Default to streaming for pre-tokenized data to avoid caching issues
        streaming_dataset = True
        logger.info("Streaming enabled by default for pre-tokenized datasets (use --force_no_streaming to disable)")
    else:
        # C4 dataset always uses streaming
        streaming_dataset = True
        logger.info("Streaming enabled for C4 dataset")
    
    train_dataset, eval_dataset, tokenizer = get_dataset(
        tokenizer_path=args.tokenizer_path,
        block_size=args.block_size,
        streaming=streaming_dataset,
        pre_tokenized_path=args.pre_tokenized_dataset_path,
        streaming_eval_samples=args.streaming_eval_samples,
        metadata_file=args.metadata_file,
        cache_dir=args.dataset_cache_dir,
        use_cache=args.use_dataset_cache,
        is_main_process=accelerator.is_main_process,
        accelerator=accelerator
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
    
    # Adjust vocab size to accommodate special tokens like pad token
    # If pad_token_id >= vocab_size, we need to expand the vocab
    required_vocab_size = tokenizer.vocab_size
    if tokenizer.pad_token_id is not None and tokenizer.pad_token_id >= required_vocab_size:
        required_vocab_size = tokenizer.pad_token_id + 1
        logger.info(
            f"Adjusting vocab_size from {tokenizer.vocab_size} to {required_vocab_size} "
            f"to accommodate pad_token_id={tokenizer.pad_token_id}"
        )
    
    if config.vocab_size != required_vocab_size:
        logger.warning(
            f"Config vocab_size ({config.vocab_size}) does not match required vocab_size ({required_vocab_size}). "
            f"Setting config.vocab_size to {required_vocab_size}."
        )
        config.vocab_size = required_vocab_size

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

    # W&B project name will be passed to TrainingArguments
    # The Trainer will handle wandb initialization on rank 0 only
    if args.report_to == "wandb":
        if os.environ.get("WANDB_DISABLED") == "true":
            logger.warning("WANDB_DISABLED=true detected, disabling wandb reporting")
            args.report_to = "none"
        else:
            try:
                import wandb
                logger.info(f"W&B logging will be enabled with project: {args.wandb_project}")
            except ImportError:
                logger.error("wandb is not installed but --report_to wandb was specified. Install with: pip install wandb")
                logger.error(f"Python path: {sys.executable}")
                logger.error(f"Python version: {sys.version}")
                raise ImportError("wandb is required when --report_to wandb is specified. Install with: pip install wandb")
    
    # Calculate max_steps if needed for streaming datasets
    if streaming_dataset and args.max_steps is None:
        if args.total_train_samples:
            # Calculate based on provided total samples
            world_size = accelerator.num_processes
            batch_size_per_step = args.per_device_train_batch_size * world_size * args.gradient_accumulation_steps
            calculated_max_steps = (args.total_train_samples // batch_size_per_step) * args.num_train_epochs
            logger.info(f"Calculated max_steps={calculated_max_steps} based on {args.total_train_samples} samples")
            args.max_steps = calculated_max_steps
        elif args.pre_tokenized_dataset_path:
            # Count actual samples from parquet files
            import glob
            import pyarrow.parquet as pq
            
            parquet_files = sorted(glob.glob(os.path.join(args.pre_tokenized_dataset_path, "*.parquet")))
            num_files = len(parquet_files)
            
            if num_files > 0:
                # Check first file for typical size
                first_table = pq.read_table(parquet_files[0])
                first_count = len(first_table)
                
                # Check last file which might be smaller
                last_table = pq.read_table(parquet_files[-1])
                last_count = len(last_table)
                
                # If all files likely have same size (except maybe the last)
                if num_files == 1:
                    total_samples = first_count
                elif last_count < first_count:
                    # Last file is partial
                    total_samples = first_count * (num_files - 1) + last_count
                else:
                    # All files likely have same size
                    total_samples = first_count * num_files
                
                logger.info(
                    f"Counted {total_samples:,} samples from {num_files} parquet files "
                    f"(first file: {first_count:,}, last file: {last_count:,})"
                )
                
                world_size = accelerator.num_processes
                batch_size_per_step = args.per_device_train_batch_size * world_size * args.gradient_accumulation_steps
                calculated_max_steps = (total_samples // batch_size_per_step) * args.num_train_epochs
                logger.info(f"Calculated max_steps={calculated_max_steps}")
                args.max_steps = calculated_max_steps
            else:
                raise ValueError(f"No parquet files found in {args.pre_tokenized_dataset_path}")
        else:
            # Default for streaming without known size (can be adjusted)
            default_steps_per_epoch = 100000  # Default assumption
            args.max_steps = default_steps_per_epoch * args.num_train_epochs
            logger.warning(
                f"Streaming dataset without known size. Using default max_steps={args.max_steps}. "
                f"Provide --total_train_samples or --max_steps for accurate training length."
            )
    
    # TrainingArguments
    logger.info("Setting up TrainingArguments...")
    training_args_dict = {
        "output_dir": args.output_dir,
        "num_train_epochs": args.num_train_epochs if args.max_steps is None else 1,
        "max_steps": args.max_steps if args.max_steps is not None else -1,
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
        "optim": "adamw_torch",
        "adam_beta1": 0.9,
        "adam_beta2": 0.95,
        "adam_epsilon": 1e-8,
        # Distributed training settings
        "ddp_find_unused_parameters": False,  # Better performance when all parameters are used
        "ddp_bucket_cap_mb": 25,  # Bucket size for DDP gradient synchronization
    }
    
    # Add wandb project name if using wandb
    if args.report_to == "wandb":
        # Use provided run name or generate one
        if args.wandb_run_name:
            training_args_dict["run_name"] = args.wandb_run_name
        else:
            training_args_dict["run_name"] = f"{args.wandb_project}-{args.model_size_name}"
        # Set wandb project via environment variable (Trainer will read this)
        os.environ["WANDB_PROJECT"] = args.wandb_project
        # Disable wandb on non-main processes
        if not accelerator.is_main_process:
            os.environ["WANDB_DISABLED"] = "true"
    
    # Handle warmup_steps vs warmup_ratio
    if args.warmup_steps is not None:
        training_args_dict["warmup_steps"] = args.warmup_steps
    else:
        training_args_dict["warmup_ratio"] = args.warmup_ratio
    
    # Add learning rate scheduler type
    training_args_dict["lr_scheduler_type"] = args.lr_scheduler_type
    
    training_args = TrainingArguments(**training_args_dict)

    # Trainer
    logger.info("Initializing Trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        processing_class=tokenizer,
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
