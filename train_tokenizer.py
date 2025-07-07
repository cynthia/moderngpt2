import argparse
import os
import shutil
import json
import sentencepiece as spm
from transformers.utils import logging

logger = logging.get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Train a SentencePiece BPE Tokenizer from text file.")
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="Path to the input text file for training the tokenizer.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="trained_tokenizer_spm",
        help="Path to save the trained tokenizer files. Will create a directory if it doesn't exist.",
    )
    parser.add_argument(
        "--vocab_size", type=int, default=32000, help="Vocabulary size for the tokenizer."
    )
    parser.add_argument(
        "--special_tokens",
        type=str,
        nargs='+',
        default=[
            "<|endoftext|>", "<unk>", "<pad>",
            # Instruction fine-tuning tokens
            "<|system|>", "<|user|>", "<|assistant|>",
            "<|begin_of_text|>", "<|end_of_text|>",
            "<|start_header_id|>", "<|end_header_id|>",
            "<|eot_id|>",  # End of turn
            "<|reserved_special_token_0|>", "<|reserved_special_token_1|>",
            "<|reserved_special_token_2|>", "<|reserved_special_token_3|>",
        ],
        help="List of special tokens to add to the tokenizer. Includes common tokens and instruction fine-tuning tokens."
    )
    parser.add_argument(
        "--character_coverage",
        type=float,
        default=0.98,
        help="Character coverage for SentencePiece. Default is 0.98 (98%)",
    )
    parser.add_argument(
        "--max_sentence_length",
        type=int,
        default=16384,
        help="Maximum sentence length for SentencePiece training",
    )
    parser.add_argument(
        "--num_threads",
        type=int,
        default=8,
        help="Number of threads to use for training",
    )

    args = parser.parse_args()

    # Setup logging
    logging.set_verbosity_info()

    # Check if input file exists
    if not os.path.exists(args.input_file):
        logger.error(f"Input file {args.input_file} does not exist.")
        return

    logger.info(f"Training tokenizer from {args.input_file}")
    
    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
        logger.info(f"Created output directory: {args.output_path}")
    
    # Prepare SentencePiece model prefix
    model_prefix = os.path.join(args.output_path, "spm")
    
    # Build SentencePiece training arguments
    spm_train_args = [
        f"--input={args.input_file}",
        f"--model_prefix={model_prefix}",
        f"--vocab_size={args.vocab_size}",
        "--model_type=bpe",
        f"--character_coverage={args.character_coverage}",
        f"--max_sentence_length={args.max_sentence_length}",
        "--pad_id=0",
        "--unk_id=1",
        "--bos_id=-1",  # Don't use BOS
        "--eos_id=2",
        "--pad_piece=<pad>",
        "--unk_piece=<unk>",
        "--normalization_rule_name=identity",  # No normalization to preserve original text
        "--byte_fallback=true",  # Enable byte fallback for out-of-vocabulary characters
        f"--num_threads={args.num_threads}",
        "--train_extremely_large_corpus=true",  # Enable training on extremely large corpus
    ]
    
    # Add user-defined special tokens.
    # Filter out tokens that SPM handles via specific parameters
    spm_reserved_tokens = {"<unk>", "<pad>"}

    filtered_user_defined_symbols = []
    if args.special_tokens:
        for token in args.special_tokens:
            if token not in spm_reserved_tokens:
                filtered_user_defined_symbols.append(token)

    if filtered_user_defined_symbols:
        # SentencePiece expects a comma-separated string for user_defined_symbols
        user_defined_symbols_str = ",".join(filtered_user_defined_symbols)
        spm_train_args.append(f"--user_defined_symbols={user_defined_symbols_str}")
    
    # Train the model
    logger.info("Starting SentencePiece tokenizer training...")
    spm.SentencePieceTrainer.train(" ".join(spm_train_args))
    logger.info("SentencePiece tokenizer training finished.")
    
    # Load the trained SentencePiece model
    sp = spm.SentencePieceProcessor()
    sp.load(f"{model_prefix}.model")
    
    # Create HuggingFace-compatible tokenizer configuration
    logger.info("Creating HuggingFace-compatible tokenizer configuration...")
    
    # Copy the SentencePiece model file to the output directory with standard name
    target_model_path = os.path.join(args.output_path, "tokenizer.model")
    shutil.copy2(f"{model_prefix}.model", target_model_path)
    logger.info(f"Copied SentencePiece model to {target_model_path}")
    
    # Create tokenizer configuration
    from moderngpt2.tokenization_moderngpt2 import ModernGPT2Tokenizer
    
    # Prepare special token mapping for tokenizer config
    special_token_mapping = {}
    added_tokens_decoder = {}
    additional_special_tokens = []
    
    # Standard tokens
    standard_tokens = {
        "<pad>": sp.piece_to_id("<pad>"),
        "<unk>": sp.piece_to_id("<unk>"),
        "<|endoftext|>": sp.piece_to_id("</s>")  # Use EOS token as fallback
    }
    
    for token, token_id in standard_tokens.items():
        if token_id != sp.unk_id():  # Only add if token exists in vocabulary
            added_tokens_decoder[str(token_id)] = {
                "content": token,
                "lstrip": False,
                "normalized": False,
                "rstrip": False,
                "single_word": False,
                "special": True
            }
    
    # Additional special tokens
    if args.special_tokens:
        for token in args.special_tokens:
            if token not in ["<unk>", "<pad>", "<|endoftext|>"]:
                token_id = sp.piece_to_id(token)
                if token_id != sp.unk_id():  # Only add if token exists in vocabulary
                    added_tokens_decoder[str(token_id)] = {
                        "content": token,
                        "lstrip": False,
                        "normalized": False,
                        "rstrip": False,
                        "single_word": False,
                        "special": True
                    }
                    additional_special_tokens.append(token)
    
    # Create tokenizer_config.json
    tokenizer_config = {
        "added_tokens_decoder": added_tokens_decoder,
        "additional_special_tokens": additional_special_tokens,
        "clean_up_tokenization_spaces": False,
        "eos_token": "<|endoftext|>",
        "model_max_length": 1024,
        "pad_token": "<pad>",
        "tokenizer_class": "ModernGPT2Tokenizer",
        "unk_token": "<unk>"
    }
    
    # Save tokenizer config
    with open(os.path.join(args.output_path, "tokenizer_config.json"), "w") as f:
        json.dump(tokenizer_config, f, indent=2)
    
    # Create special_tokens_map.json
    special_tokens_map = {
        "eos_token": "<|endoftext|>",
        "pad_token": "<pad>",
        "unk_token": "<unk>"
    }
    
    with open(os.path.join(args.output_path, "special_tokens_map.json"), "w") as f:
        json.dump(special_tokens_map, f, indent=2)
    
    logger.info(f"Tokenizer saved to {args.output_path} (compatible with ModernGPT2Tokenizer)")
    logger.info(f"Special tokens included: {args.special_tokens}")
    logger.info(f"Vocabulary size: {sp.get_piece_size()}")
    
    # Copy the ModernGPT2Tokenizer class definition file to the output directory
    tokenizer_class_def_source_file = os.path.join(os.path.dirname(__file__), "moderngpt2", "tokenization_moderngpt2.py")
    tokenizer_class_def_dest_file = os.path.join(args.output_path, "custom_tokenizer_code.py")

    if os.path.exists(tokenizer_class_def_source_file):
        shutil.copy2(tokenizer_class_def_source_file, tokenizer_class_def_dest_file)
        logger.info(f"Copied tokenizer class definition for trust_remote_code support.")
    else:
        logger.warning(f"Tokenizer class definition file '{tokenizer_class_def_source_file}' not found.")

    logger.info(f"Tokenizer training complete. Files saved in {args.output_path}")
    logger.info(f"To use this tokenizer, point --tokenizer_path to '{args.output_path}'")


if __name__ == "__main__":
    main()