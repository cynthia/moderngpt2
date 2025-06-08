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

### Step 3: Train Your Model

We support various model sizes (small, medium, large, xl) and hardware configurations:

#### Single GPU Training

```bash
deepspeed train.py \
    --deepspeed \
    --deepspeed_config "ds_config_zero1.json" \
    --model_size_name "medium" \
    --tokenizer_path "./my_tokenizer" \
    --output_dir "output/model" \
    --num_train_epochs 3 \
    --per_device_train_batch_size 32 \
    --learning_rate 2.5e-4
```

#### Multi-GPU Training

```bash
accelerate launch --config_file accelerate_config.yaml --num_processes 4 train.py \
    --model_size_name "medium" \
    --tokenizer_path "./my_tokenizer" \
    --pre_tokenized_dataset_path "./my_pretokenized_data" \
    --output_dir "output/multi_gpu_model" \
    --deepspeed_config "ds_config_zero1.json"
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