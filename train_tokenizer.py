import argparse
import os
import tempfile
from datasets import load_dataset, interleave_datasets
import sentencepiece as spm
from transformers import PreTrainedTokenizerFast
from transformers.utils import logging
from tokenizers import Tokenizer

logger = logging.get_logger(__name__)

def dataset_text_iterator(dataset, batch_size=1000):
    """
    An iterator that yields batches of text from a Hugging Face dataset.
    Specifically designed for datasets where each item has a 'text' field.
    """
    batch = []
    for example in dataset:
        text = example.get("text")
        if text and isinstance(text, str): # Ensure text exists and is a string
            batch.append(text)
            if len(batch) == batch_size:
                yield batch
                batch = []
    if batch: # Yield any remaining texts
        yield batch

def main():
    parser = argparse.ArgumentParser(description="Train a SentencePiece Unigram Tokenizer on C4 dataset.")
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
        "--max_train_lines",
        type=int,
        default=1000000, # 1 million lines from C4 as a default for training
        help="Maximum number of lines from C4 to use for training the tokenizer. Streams data.",
    )
    parser.add_argument(
        "--text_iterator_batch_size",
        type=int,
        default=1000,
        help="Batch size for yielding text to the tokenizer trainer.",
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
        default=0.9995,
        help="Character coverage for SentencePiece. Default is 0.9995 (99.95%)",
    )
    parser.add_argument(
        "--max_sentence_length",
        type=int,
        default=16384,
        help="Maximum sentence length for SentencePiece training",
    )

    args = parser.parse_args()

    # Setup logging
    logging.set_verbosity_info()

    logger.info("Loading and preparing C4 dataset for tokenizer training...")
    langs = ["en", "ja", "ko", "zh"]

    datasets_list_processed = []
    for lang in langs:
        logger.info(f"Loading C4/{lang} for tokenizer training...")
        dset = load_dataset("allenai/c4", lang, streaming=True, split="train", trust_remote_code=True)

        columns_to_remove = []
        try:
            # Attempt to get column names from features if available
            current_columns = list(dset.features.keys())
            columns_to_remove = [col for col in current_columns if col != 'text']
        except Exception as e:
            logger.warning(f"Could not dynamically determine columns for C4/{lang} due to: {e}. "
                           "Falling back to a predefined list of common columns to remove.")
            # Common columns in C4 that are not 'text'
            columns_to_remove = ['timestamp', 'url', 'c4_language', 'metadata']
            columns_to_remove = [col for col in columns_to_remove if col != 'text']

        if columns_to_remove:
            logger.info(f"For C4/{lang}, attempting to remove columns: {columns_to_remove} to keep only 'text' for feature alignment.")
            dset = dset.map(lambda x: x, batched=True, remove_columns=columns_to_remove)

        datasets_list_processed.append(dset)

    logger.info("Interleaving datasets...")
    interleaved_ds = interleave_datasets(datasets_list_processed)

    logger.info(f"Taking approx {args.max_train_lines} lines for tokenizer training.")
    training_data_subset = interleaved_ds.take(args.max_train_lines)

    # Create a temporary file to store training data for SentencePiece
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as tmp_file:
        tmp_filename = tmp_file.name
        logger.info(f"Writing training data to temporary file: {tmp_filename}")
        
        line_count = 0
        for batch in dataset_text_iterator(training_data_subset, batch_size=args.text_iterator_batch_size):
            for text in batch:
                tmp_file.write(text + '\n')
                line_count += 1
                if line_count >= args.max_train_lines:
                    break
            if line_count >= args.max_train_lines:
                break
        
        logger.info(f"Wrote {line_count} lines to temporary file")

    # Train SentencePiece model
    logger.info("Starting SentencePiece tokenizer training...")
    
    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
        logger.info(f"Created output directory: {args.output_path}")
    
    # Prepare SentencePiece model prefix
    model_prefix = os.path.join(args.output_path, "spm")
    
    # Build SentencePiece training arguments
    spm_train_args = [
        f"--input={tmp_filename}",
        f"--model_prefix={model_prefix}",
        f"--vocab_size={args.vocab_size}",
        "--model_type=unigram",
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
    ]
    
    # Add user-defined special tokens.
    # These tokens are added to the vocabulary by SentencePiece directly.
    # Filter out tokens that SPM handles via specific parameters (e.g. <unk>)
    # or that might cause issues if passed explicitly as user_defined_symbols.
    spm_reserved_tokens = {"<unk>"}  # Add other SPM-specific tokens like <s>, </s>, <pad> if they are managed by specific params

    filtered_user_defined_symbols = []
    if args.special_tokens:
        for token in args.special_tokens:
            if token not in spm_reserved_tokens:
                filtered_user_defined_symbols.append(token)

    if filtered_user_defined_symbols:
        # SentencePiece expects a comma-separated string for user_defined_symbols.
        # It's important that args.vocab_size is set large enough to accommodate these
        # in addition to the tokens learned from data.
        user_defined_symbols_str = ",".join(filtered_user_defined_symbols)
        spm_train_args.append(f"--user_defined_symbols={user_defined_symbols_str}")
    
    # Train the model
    spm.SentencePieceTrainer.train(" ".join(spm_train_args))
    logger.info("SentencePiece tokenizer training finished.")
    
    # Clean up temporary file
    os.unlink(tmp_filename)
    logger.info(f"Cleaned up temporary file: {tmp_filename}")
    
    # Load the trained SentencePiece model
    sp = spm.SentencePieceProcessor()
    sp.load(f"{model_prefix}.model")
    
    # Create HuggingFace-compatible tokenizer using the custom ModernGPT2Tokenizer
    logger.info("Creating HuggingFace-compatible tokenizer configuration...")
    
    # Copy the SentencePiece model file to the output directory with standard name
    import shutil
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
        "tokenizer_class": "ModernGPT2Tokenizer",  # Just the class name
        "unk_token": "<unk>"
        # auto_map is removed.
    }
    
    # Save tokenizer config
    import json
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
    
    # Also keep the original SentencePiece files
    logger.info(f"Original SentencePiece model files: {model_prefix}.model and {model_prefix}.vocab")
    
    logger.info(f"Tokenizer training and saving complete. Files are in {args.output_path}")
    logger.info(f"To use this tokenizer with train.py or dataset.py, point --tokenizer_path to '{args.output_path}'")

    logger.info(f"Tokenizer training and saving complete. Files are in {args.output_path}")
    logger.info(f"To use this tokenizer with train.py or dataset.py, point --tokenizer_path to '{args.output_path}'")

    # Copy the ModernGPT2Tokenizer class definition file to the output directory
    # and rename it for clarity with auto_map.
    tokenizer_class_def_source_file = os.path.join(os.path.dirname(__file__), "moderngpt2", "tokenization_moderngpt2.py")
    # The destination filename *must* be what's referenced in auto_map (e.g., custom_tokenizer_code.py)
    tokenizer_class_def_dest_file = os.path.join(args.output_path, "custom_tokenizer_code.py") # Keep this name for the .py file

    if os.path.exists(tokenizer_class_def_source_file):
        shutil.copy2(tokenizer_class_def_source_file, tokenizer_class_def_dest_file)
        logger.info(f"Copied '{tokenizer_class_def_source_file}' to {tokenizer_class_def_dest_file} for trust_remote_code support.")

        # Remove __init__.py creation if it exists from previous attempts
        init_py_path = os.path.join(args.output_path, "__init__.py")
        if os.path.exists(init_py_path):
            os.remove(init_py_path)
            logger.info(f"Removed '{init_py_path}' if it existed.")
    else:
        logger.warning(f"Tokenizer class definition file '{tokenizer_class_def_source_file}' not found. Cannot copy for trust_remote_code.")

if __name__ == "__main__":
    main()