import argparse
import os
import datasets
from datasets import load_dataset, interleave_datasets, Dataset
from transformers import AutoTokenizer # Using AutoTokenizer for flexibility
from transformers.utils import logging
import pyarrow.parquet as pq
import pyarrow as pa

logger = logging.get_logger(__name__)
datasets.disable_caching() # Disable caching for streaming, good practice for large datasets

def main():
    parser = argparse.ArgumentParser(description="Pre-tokenize a dataset and save to Parquet.")
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        required=True,
        help="Path to the trained tokenizer directory (containing tokenizer.json).",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Directory to save the pre-tokenized Parquet files.",
    )
    parser.add_argument(
        "--block_size", type=int, default=1024, help="Block size for grouping tokenized texts."
    )
    parser.add_argument(
        "--max_samples_per_shard",
        type=int,
        default=200000, # Number of tokenized+grouped samples per Parquet file
        help="Maximum number of processed samples (blocks) to save in each Parquet shard.",
    )
    parser.add_argument(
        "--dataset_streaming",
        action="store_true",
        help="Stream the C4 dataset. Good for memory efficiency.",
    )
    parser.add_argument(
        "--no_dataset_streaming",
        action="store_false",
        dest="dataset_streaming",
        help="Do not stream C4 (loads everything into memory - not recommended for full C4).",
    )
    parser.set_defaults(dataset_streaming=True)
    parser.add_argument(
        "--c4_langs",
        type=str,
        nargs='+',
        default=["en", "ja", "ko", "zh"],
        help="List of C4 languages to process."
    )
    parser.add_argument(
        "--max_input_lines_total",
        type=int,
        default=None, # Process all available lines by default
        help="Total maximum number of raw text lines from C4 to process for pre-tokenization. Spans across all shards."
    )


    args = parser.parse_args()

    logging.set_verbosity_info()

    logger.info(f"Loading tokenizer from: {args.tokenizer_path}")
    # Use AutoTokenizer which can load ModernGPT2TokenizerFast or other tokenizer types
    # This assumes tokenizer_path is a directory containing tokenizer.json
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True) # Added trust_remote_code for custom tokenizers
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token # Ensure pad token is set

    logger.info(f"Loading C4 dataset for languages: {args.c4_langs} (Streaming: {args.dataset_streaming})")

    # Load multiple language datasets
    raw_datasets_list = [
        load_dataset("allenai/c4", lang, streaming=args.dataset_streaming, split="train", trust_remote_code=True)
        for lang in args.c4_langs
    ]
    raw_dataset = interleave_datasets(raw_datasets_list)

    if args.max_input_lines_total is not None:
        logger.info(f"Limiting total input lines to process to: {args.max_input_lines_total}")
        raw_dataset = raw_dataset.take(args.max_input_lines_total)


    def tokenize_function(examples):
        texts = examples["text"]
        # Filter out None or empty strings if any slip through, though map might handle some.
        valid_texts = [t for t in texts if t and isinstance(t, str) and t.strip()]
        if not valid_texts:
            return {"input_ids": [], "attention_mask": []}
        # Tokenize. `truncation=False` because grouping will handle fixed lengths.
        output = tokenizer(valid_texts, truncation=False)
        return output

    logger.info("Mapping tokenization function over the dataset...")
    # Batched=True is important for efficiency. Batch size for map can be default or tuned.
    tokenized_dataset = raw_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text", "timestamp", "url", "c4_language"] # Remove original and any lang-specific columns
    )

    # Grouping into Blocks
    def group_texts(examples):
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])

        if total_length == 0:
            return {k: [] for k in examples.keys()} # Return empty if no data

        # We drop the small remainder.
        if total_length >= args.block_size:
            total_length = (total_length // args.block_size) * args.block_size
        else: # If total length is less than block_size, effectively no full blocks
            return {k: [] for k in examples.keys()} # Return empty

        result = {
            k: [t[i : i + args.block_size] for i in range(0, total_length, args.block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    logger.info(f"Mapping grouping function (block_size={args.block_size}) over the tokenized dataset...")
    # Batched=True for group_texts as well.
    # The batch size for this map can also be tuned if needed, default is 1000.
    # Filter out empty results from tokenize_function or group_texts before attempting to save.
    lm_dataset = tokenized_dataset.map(group_texts, batched=True).filter(lambda x: len(x['input_ids']) > 0)

    os.makedirs(args.output_path, exist_ok=True)
    logger.info(f"Output directory set to: {args.output_path}")

    # Iterate, save to Parquet in shards
    shard_index = 0
    processed_samples_in_current_shard = 0
    current_shard_data = {k: [] for k in ["input_ids", "attention_mask", "labels"]}

    logger.info(f"Starting processing and sharding. Max samples per shard: {args.max_samples_per_shard}")

    for processed_example in lm_dataset: # Iterate through the final processed dataset
        if not processed_example['input_ids']: # Should be filtered out by .filter, but as a safeguard
            continue

        current_shard_data["input_ids"].append(processed_example["input_ids"])
        current_shard_data["attention_mask"].append(processed_example["attention_mask"])
        current_shard_data["labels"].append(processed_example["labels"])
        processed_samples_in_current_shard += 1

        if processed_samples_in_current_shard >= args.max_samples_per_shard:
            shard_file_path = os.path.join(args.output_path, f"shard_{shard_index}.parquet")
            try:
                table = pa.Table.from_pydict(current_shard_data)
                pq.write_table(table, shard_file_path)
                logger.info(f"Saved shard {shard_index} to {shard_file_path} with {processed_samples_in_current_shard} samples.")
            except Exception as e:
                logger.error(f"Error writing shard {shard_index} to {shard_file_path}: {e}")

            shard_index += 1
            processed_samples_in_current_shard = 0
            current_shard_data = {k: [] for k in ["input_ids", "attention_mask", "labels"]}

    # Save any remaining data in the last shard
    if processed_samples_in_current_shard > 0:
        shard_file_path = os.path.join(args.output_path, f"shard_{shard_index}.parquet")
        try:
            table = pa.Table.from_pydict(current_shard_data)
            pq.write_table(table, shard_file_path)
            logger.info(f"Saved final shard {shard_index} to {shard_file_path} with {processed_samples_in_current_shard} samples.")
        except Exception as e:
            logger.error(f"Error writing final shard {shard_index} to {shard_file_path}: {e}")

    logger.info("Dataset pre-tokenization and sharding complete.")
    if shard_index == 0 and processed_samples_in_current_shard == 0:
        logger.warning("No data was processed or saved. Check dataset source or processing parameters (e.g. max_input_lines_total, block_size).")

if __name__ == "__main__":
    main()
