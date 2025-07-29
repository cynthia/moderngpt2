# ModernGPT2

A modern reimplementation of GPT-2 that incorporates architectural improvements from recent language models like Gemma2. This project demonstrates how classic transformer architectures can benefit from modern techniques while maintaining compatibility with the Hugging Face ecosystem.

## Architecture Improvements

ModernGPT2 enhances the original GPT-2 architecture with several key improvements:

### 1. RMSNorm Instead of LayerNorm
We've replaced the standard LayerNorm with Root Mean Square Normalization (RMSNorm). This simplification normalizes based on the RMS statistics and uses a learnable scaling parameter initialized to zeros, making training more stable.

### 2. Rotary Position Embeddings (RoPE)
Gone are the learned absolute position embeddings. RoPE injects positional information directly into the attention mechanism through rotation of query and key vectors, enabling better length generalization.

### 3. Grouped Query Attention (GQA)
The attention mechanism now supports using fewer key/value heads than query heads. When `num_key_value_heads < n_head`, keys and values are shared across multiple query heads, significantly reducing memory usage while maintaining performance.

### 4. Gated MLP (GeGLU variant)
The feed-forward network uses a gating mechanism with three projections:
- Gate projection with activation function
- Up projection for feature expansion
- Down projection after element-wise multiplication

This allows for more sophisticated feature interactions compared to the standard two-layer MLP.

### 5. Enhanced Stability Features
- **Query scaling**: Queries are scaled by `head_dim^-0.5` by default
- **Attention softcapping**: Optional logit capping before softmax to prevent training instabilities
- **Embedding scaling**: Input embeddings are scaled by `sqrt(hidden_size)`

### 6. Modernized Block Structure
Each transformer block now follows a more sophisticated residual pattern with normalization both before and after each major operation (attention and MLP), improving gradient flow.

## Getting Started

### Requirements

1. **Tokenizer**: You'll need a SentencePiece or Hugging Face tokenizer. Train one from scratch using our `train_tokenizer.py` script or bring your own.

2. **Dataset**: Two options are available:
   - Stream the multilingual C4 dataset directly (requires internet)
   - Pre-tokenize the data for faster training using `pretokenize_dataset.py`

3. **Hardware**: Designed for GPU training with DeepSpeed. We include a ZeRO Stage 1 configuration that works well from single GPU setups to multi-node clusters.

## Quick Start

The training pipeline consists of three main steps:

### Step 1: Train Your Tokenizer

If you don't have a tokenizer, train one on multilingual C4 data:

```bash
python train_tokenizer.py \
    --output_path ./my_tokenizer \
    --vocab_size 32000 \
    --max_train_lines 1000000 \
    --special_tokens "<|endoftext|>" "<unk>" "<pad>"
```

This trains a 32K vocabulary on 1M lines from C4 (supports en, ja, ko, zh).

### Step 2: Pre-tokenize Your Data (Optional but Recommended)

Pre-tokenizing avoids redundant tokenization during training:

```bash
python pretokenize_dataset.py \
    --tokenizer_path ./my_tokenizer \
    --output_path ./my_pretokenized_data \
    --block_size 1024 \
    --max_samples_per_shard 200000 \
    --c4_langs "en" "ja" \
    --max_input_lines_total 5000000
```

Processes 5M lines from English and Japanese C4, creating sharded Parquet files with 1024-token blocks.

**Note on Parallelism in Pre-tokenization:**
- The `pretokenize_dataset.py` script can utilize multiple CPU cores for faster processing using the `--num_proc <number>` argument. By default, it uses (CPU count - 2) processes.
- However, **parallel processing via `--num_proc` is only effective if you run the script with the `--no_dataset_streaming` flag.** This is because the `datasets.map()` function cannot use multiprocessing with streaming datasets.
- **Warning:** Using `--no_dataset_streaming` will cause the script to download and load the *entirety* of the specified C4 language splits into your Hugging Face cache directory (or memory if cache is disabled) before processing begins. This can require a very large amount of disk space and memory, especially for multiple languages from C4.
- If you use the default `--dataset_streaming` (or explicitly specify it), pre-tokenization will run on a single core but will be more memory-efficient for very large datasets.

## BitBPE: Enhanced Tokenization for Multilingual Models

ModernGPT2 now supports BitBPE (Bit-level Byte Pair Encoding), an alternative UTF-8 byte representation for multilingual models, particularly those handling CJK (Chinese, Japanese, Korean) languages.

### What is BitBPE?

BitBPE is based on the paper ["Bit-level BPE: Below the byte boundary"](https://arxiv.org/abs/2506.07541). It provides an alternative representation for UTF-8 byte sequences:

- **Standard UTF-8**: Treats each UTF-8 byte as an atomic 8-bit unit
- **UTF-8 in BitBPE**: Breaks the 8-bit boundary, using 6-bit prefixes and 9-bit tokens for better compression

### Benefits for CJK Languages

CJK characters typically require 3 bytes in UTF-8 encoding. BitBPE reduces this overhead by:
- Using a 6-bit prefix token for common UTF-8 patterns
- Redistributing the remaining 18 bits into two 9-bit tokens
- Achieving up to 22% reduction in sequence length for byte-fallback encoded CJK text

### Converting to BitBPE

Convert an existing BPE tokenizer to BitBPE format:

```bash
python convert_bpe_to_bitbpe.py \
    --input_path model/bpe-8k \
    --output_path model/bpe-8k-bitbpe \
    --num_prefix_tokens 64 \
    --test_conversion
```

### Pre-tokenizing with BitBPE

Use the streaming pre-tokenization script with BitBPE:

```bash
python pretokenize_dataset_streaming.py \
    --tokenizer_path model/bpe-8k-bitbpe \
    --output_path data/pretokenized-bitbpe \
    --tokenizer_type bitbpe \
    --languages en ja ko \
    --max_samples_total 5000000 \
    --block_size 1024
```

Key parameters:
- `--tokenizer_type bitbpe`: Enables BitBPE encoding
- `--languages`: Supports en (English/C4), ja (Japanese/FineWeb2), ko (Korean/FineWeb2)
- `--max_samples_total`: Samples per language (5M = ~5B tokens with 1024 block size)

### Training with BitBPE Data

Training with BitBPE pre-tokenized data is identical to standard training:

```bash
acccelerate launch --config_file accelerate_config.yaml --num_processes 4 train.py \
    --model_size_name "small" \
    --tokenizer_path "model/bpe-8k-bitbpe" \
    --pre_tokenized_dataset_path "data/pretokenized-bitbpe" \
    --output_dir "output/bitbpe-model" \
    --num_train_epochs 1 \
    --per_device_train_batch_size 32 \
    --learning_rate 2.5e-4
```

### Technical Details

BitBPE implements the bit redistribution algorithm for 3-byte UTF-8 sequences (0xE0-0xEF):

1. **Original**: `[b1: 8 bits] [b2: 8 bits] [b3: 8 bits]` (24 bits total)
2. **BitBPE**: `[b1': 6 bits] [b2': 9 bits] [b3': 9 bits]` (24 bits total)

The transformation:
- b1' (prefix): `b1 >> 2` (top 6 bits of first byte)
- b2': `((b1 & 3) << 7) | ((b2 & 254) >> 1)`
- b3': `((b2 & 1) << 8) | b3`

To reduce redundancy, repeated prefixes are deduplicated - if consecutive 3-byte sequences share the same prefix (b1'), the prefix token is only emitted once.

This maintains the same information while allowing the tokenizer to learn more efficient representations for common CJK patterns.

### Step 3: Train Your Model

We support various model sizes (small, medium, large, xl) and hardware configurations:

The training script offers two main ways to handle your dataset:
*   **Using pre-tokenized data (recommended):** Provide the path to your pre-processed Parquet files using the `--pre_tokenized_dataset_path /path/to/your/pretokenized_data` argument. This is generally faster as tokenization is done only once. You can generate this data using the `pretokenize_dataset.py` script.
*   **On-the-fly C4 streaming:** If `--pre_tokenized_dataset_path` is omitted, the script defaults to streaming the C4 dataset and tokenizing it during training. In this mode, ensure you set `--block_size` (e.g., `--block_size 1024`) to define the sequence length for the model.

**Mixed Precision Training:**
- Use `--fp16` to enable FP16 mixed precision training.
- Use `--bf16` to enable BF16 mixed precision training (requires Ampere or newer NVIDIA GPUs, or compatible hardware).
- Note: `--fp16` and `--bf16` are mutually exclusive.

#### Single GPU Training

```bash
deepspeed train.py \
    --deepspeed \
    --deepspeed_config "ds_config_zero1.json" \
    --model_size_name "medium" \
    --tokenizer_path "./my_tokenizer" \
    --pre_tokenized_dataset_path "./my_pretokenized_data" \
    --output_dir "output/model" \
    --num_train_epochs 3 \
    --per_device_train_batch_size 16 \
    --learning_rate 2.5e-4 \
    --ds_config "ds_config_zero1.json" \
    --fp16 \
    --gradient_accumulation_steps 1
```

#### Multi-GPU Training

```bash
accelerate launch --config_file accelerate_config.yaml --num_processes 4 train.py \
    --model_size_name "medium" \
    --tokenizer_path "./my_tokenizer" \
    --pre_tokenized_dataset_path "./my_pretokenized_data" \
    --output_dir "output/multi_gpu_model" \
    --ds_config "ds_config_zero1.json" \
    --num_train_epochs 3 \
    --per_device_train_batch_size 16 \
    --learning_rate 2.5e-4 \
    --fp16 \
    --gradient_accumulation_steps 1
```

> **Note**: First run `accelerate config` to create your configuration file. The provided `accelerate_config.yaml` assumes 4 GPUs with DeepSpeed.

For detailed training examples on specific hardware (TSUBAME 4, ABCI 3.0, etc.), see [training_examples.md](training_examples.md).

## Configuration Details

ModernGPT2 includes several new configuration parameters:

- `num_key_value_heads`: For grouped query attention (use fewer than `n_head` to save memory)
- `rope_theta`: Base frequency for rotary embeddings (default: 10000.0)
- `attn_logit_softcapping`: Caps attention logits for stability
- `activation_function`: Supports various activations (default: "gelu_pytorch_tanh")
- `rms_norm_eps`: Epsilon for RMSNorm layers (default: 1e-6)

See `moderngpt2/configuration_moderngpt2.py` for all available options.

### BitBPE Configuration

When using a BitBPE tokenizer, the model automatically detects and handles the special encoding:

- The tokenizer configuration includes `tokenizer_type: "bitbpe"`
- Virtual vocabulary size increases by the number of prefix tokens (default: 64)
- The model seamlessly handles both standard and BitBPE tokens during training

## Hardware Recommendations

- **H100/H200 (80GB+)**: Use batch size 32-64 per device
- **A100 (40-80GB)**: Use batch size 16-32 per device  
- **A6000/RTX 4090 (24-48GB)**: Use batch size 8-16 per device

The included DeepSpeed configuration uses ZeRO Stage 1 optimization, which works well for most setups. For larger models or limited memory, consider ZeRO Stage 2 or 3.

## Development

The codebase is organized as:
- `moderngpt2/`: Core model implementation (PyTorch, TensorFlow, JAX)
- `train.py`: Main training script with Hugging Face Trainer
- `dataset.py`: Data loading utilities with streaming support
- Configuration files for DeepSpeed and Accelerate

Install dependencies:
```bash
pip install torch transformers datasets deepspeed accelerate
```

## License

This project builds upon the Hugging Face Transformers library and incorporates techniques from Google's Gemma2 model. Please refer to their respective licenses.