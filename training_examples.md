# Training Examples for Different Hardware Configurations

This document provides example training commands for various hardware setups. Adjust parameters like `--model_size_name`, `--tokenizer_path`, `--output_dir`, `--learning_rate`, `--num_train_epochs`, and batch sizes according to your specific needs.

## Key Arguments

- `--deepspeed_config ds_config_zero1.json`: Specifies the DeepSpeed configuration for ZeRO-1
- `--tokenizer_path <path>`: Path to your SentencePiece tokenizer
- `--output_dir <path>`: Where to save model checkpoints and logs
- `--model_size_name`: Choose from "small", "medium", "large", "xl"
- `--per_device_train_batch_size`: Batch size per GPU
- `--pre_tokenized_dataset_path <path>`: Path to pre-tokenized Parquet files (optional)
- `--block_size`: Token sequence length (default: 1024)

## Single GPU Configurations

### TSUBAME 4: 1x H100 (80GB)
```bash
deepspeed train.py \
    --deepspeed \
    --deepspeed_config "ds_config_zero1.json" \
    --model_size_name "medium" \
    --tokenizer_path "path/to/your/tokenizer" \
    --output_dir "output/tsubame4_1h100_zero1" \
    --num_train_epochs 3 \
    --per_device_train_batch_size 32 \
    --learning_rate 2.5e-4 \
    --report_to "wandb" \
    --block_size 1024
```

### ABCI 3.0: 1x H200 (141GB)
```bash
deepspeed train.py \
    --deepspeed \
    --deepspeed_config "ds_config_zero1.json" \
    --model_size_name "medium" \
    --tokenizer_path "path/to/your/tokenizer" \
    --output_dir "output/abci3_1h200_zero1" \
    --num_train_epochs 3 \
    --per_device_train_batch_size 64 \
    --learning_rate 2.5e-4 \
    --report_to "wandb" \
    --block_size 1024
```

### Local: 1x A6000 (48GB)
```bash
deepspeed train.py \
    --deepspeed \
    --deepspeed_config "ds_config_zero1.json" \
    --model_size_name "medium" \
    --tokenizer_path "path/to/your/tokenizer" \
    --output_dir "output/local_1a6000_zero1" \
    --num_train_epochs 3 \
    --per_device_train_batch_size 16 \
    --learning_rate 2.5e-4 \
    --report_to "wandb" \
    --block_size 1024
```

## Multi-GPU Configurations

### TSUBAME 4: 4x H100 (80GB)
```bash
accelerate launch --config_file accelerate_config.yaml --num_processes 4 train.py \
    --model_size_name "medium" \
    --tokenizer_path "./my_tokenizer" \
    --pre_tokenized_dataset_path "./my_pretokenized_data" \
    --output_dir "output/tsubame4_4h100_zero1" \
    --num_train_epochs 3 \
    --per_device_train_batch_size 32 \
    --learning_rate 2.5e-4 \
    --deepspeed_config "ds_config_zero1.json" \
    --report_to "wandb" \
    --block_size 1024
```

### ABCI 3.0: 8x H200 (141GB)
```bash
accelerate launch --config_file accelerate_config.yaml --num_processes 8 train.py \
    --model_size_name "medium" \
    --tokenizer_path "path/to/your/tokenizer" \
    --output_dir "output/abci3_8h200_zero1" \
    --num_train_epochs 3 \
    --per_device_train_batch_size 64 \
    --learning_rate 2.5e-4 \
    --deepspeed_config "ds_config_zero1.json" \
    --report_to "wandb" \
    --block_size 1024
```

## Accelerate Configuration

For multi-GPU training, you'll need an `accelerate_config.yaml`. Generate one using:
```bash
accelerate config
```

Or use this basic example for 4 GPUs with DeepSpeed:
```yaml
compute_environment: LOCAL_MACHINE
distributed_type: DEEPSPEED
num_processes: 4
machine_rank: 0
num_machines: 1
gpu_ids: all
deepspeed_config: {} # Will be overridden by --deepspeed_config in train.py
use_cpu: false
```

## Batch Size Recommendations

- **H100/H200 (80GB+)**: 32-64 per device
- **A100 (40-80GB)**: 16-32 per device
- **A6000/RTX 4090 (24-48GB)**: 8-16 per device
- **Smaller GPUs**: 4-8 per device

Adjust based on model size and sequence length.