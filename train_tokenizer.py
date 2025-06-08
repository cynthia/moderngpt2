import argparse
import os
from datasets import load_dataset, interleave_datasets
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers
from tokenizers.processors import TemplateProcessing
from transformers.utils import logging # For logger if needed

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
        default=["<|endoftext|>", "<unk>", "<pad>"], # Common special tokens
        help="List of special tokens to add to the tokenizer. <|endoftext|> is typical for GPT. <unk> for unknown, <pad> for padding."
    )


    args = parser.parse_args()

    # Setup logging
    logging.set_verbosity_info()

    logger.info("Initializing a new Tokenizer with SentencePiece Unigram model.")
    gpt2_tokenizer = Tokenizer(models.Unigram())

    gpt2_tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
    gpt2_tokenizer.decoder = decoders.ByteLevel()

    eos_token_str = args.special_tokens[0] if args.special_tokens and "<|endoftext|>" in args.special_tokens else "<|endoftext|>"
    if eos_token_str not in args.special_tokens: # Ensure EOS token is part of special_tokens list for the trainer
        # This case should ideally be handled by ensuring user includes it or has a sensible default.
        # For robustness, if <|endoftext|> is not in custom list, we add it for post-processor,
        # but this might conflict if user intends a different EOS.
        # Safest is to rely on args.special_tokens containing the intended EOS.
        # Let's assume the first token in args.special_tokens is the primary EOS for simplicity.
        eos_token_str = args.special_tokens[0] if args.special_tokens else "<|endoftext|>"


    # Ensure special_tokens are correctly mapped for TemplateProcessing
    # The ID for TemplateProcessing should be its final ID in the tokenizer after training.
    # We add special tokens to the trainer, which assigns them IDs starting from 0.
    # So, we can find the index of eos_token_str in args.special_tokens to get its presumed ID for template.
    try:
        eos_token_id = args.special_tokens.index(eos_token_str)
    except ValueError:
        # This would happen if eos_token_str (e.g. default <|endoftext|>) isn't in a custom args.special_tokens list
        logger.warning(f"EOS token '{eos_token_str}' not found in special_tokens list: {args.special_tokens}. Using ID {len(args.special_tokens)} as a fallback for template.")
        # Add it to the list for the trainer if not present, and assign it the next available ID.
        if eos_token_str not in args.special_tokens:
            args.special_tokens.append(eos_token_str)
        eos_token_id = len(args.special_tokens) -1


    gpt2_tokenizer.post_processor = TemplateProcessing(
        single=f"$A {eos_token_str}",
        pair=f"$A {eos_token_str} $B:1 {eos_token_str}:1",
        special_tokens=[
            (eos_token_str, eos_token_id)
        ]
    )

    logger.info("Loading and preparing C4 dataset for tokenizer training...")
    langs = ["en", "ja", "ko", "zh"]

    datasets_list_processed = [] # Use a new name for the processed list
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
            columns_to_remove = ['timestamp', 'url', 'c4_language', 'metadata'] # Added 'metadata' as another potential one
            # Ensure 'text' is not in the list of columns to remove, just in case.
            columns_to_remove = [col for col in columns_to_remove if col != 'text']

        if columns_to_remove:
            logger.info(f"For C4/{lang}, attempting to remove columns: {columns_to_remove} to keep only 'text' for feature alignment.")
            # The map function will ignore any column in remove_columns that doesn't actually exist in the dataset.
            dset = dset.map(lambda x: x, batched=True, remove_columns=columns_to_remove)

        datasets_list_processed.append(dset)

    logger.info("Interleaving datasets...")
    # Pass the list of processed datasets to interleave_datasets
    interleaved_ds = interleave_datasets(datasets_list_processed)

    logger.info(f"Taking approx {args.max_train_lines} lines for tokenizer training.")
    training_data_subset = interleaved_ds.take(args.max_train_lines)

    trainer = trainers.UnigramTrainer(
        vocab_size=args.vocab_size,
        special_tokens=args.special_tokens,
        unk_token="<unk>" if "<unk>" in args.special_tokens else None,
        byte_fallback=True,
    )

    logger.info("Starting tokenizer training...")
    gpt2_tokenizer.train_from_iterator(
        dataset_text_iterator(training_data_subset, batch_size=args.text_iterator_batch_size),
        trainer=trainer,
        length=args.max_train_lines
    )
    logger.info("Tokenizer training finished.")

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
        logger.info(f"Created output directory: {args.output_path}")

    tokenizer_json_path = os.path.join(args.output_path, "tokenizer.json")
    gpt2_tokenizer.save(tokenizer_json_path)
    logger.info(f"Tokenizer saved to {tokenizer_json_path}")

    logger.info(f"Tokenizer training and saving complete. Files are in {args.output_path}")
    logger.info(f"To use this tokenizer with train.py or dataset.py, point --tokenizer_path to '{args.output_path}'")

if __name__ == "__main__":
    main()
