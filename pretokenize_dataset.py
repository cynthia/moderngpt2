import argparse
import os
import datasets
from datasets import load_dataset, interleave_datasets
from functools import partial # Add this import
from moderngpt2 import ModernGPT2Tokenizer
from transformers.utils import logging
import pyarrow.parquet as pq
import pyarrow as pa
from tqdm import tqdm

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
        help="Stream the C4 dataset. Good for memory efficiency (disables multi-core parallelism for .map operations).",
    )
    parser.add_argument(
        "--no_dataset_streaming",
        action="store_false",
        dest="dataset_streaming",
        help="Do not stream C4. Loads entire dataset into memory/cache (required for multi-core parallelism with --num_proc, but uses significant resources).",
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
    parser.add_argument(
        "--num_proc",
        type=int,
        default=None,
        help="Number of processes to use for parallel tokenization and grouping (only effective if --no_dataset_streaming is used). Default: CPU count - 2"
    )


    args = parser.parse_args()
    
    # Set num_proc with CPU count - 2 as default, with a minimum of 1
    if args.num_proc is None:
        import multiprocessing
        args.num_proc = max(1, multiprocessing.cpu_count() - 2)

    logging.set_verbosity_info()

    logger.info(f"Loading tokenizer from: {args.tokenizer_path}")
    # Use AutoTokenizer which can load ModernGPT2TokenizerFast or other tokenizer types
    # This assumes tokenizer_path is a directory containing tokenizer.json
    tokenizer = ModernGPT2Tokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True) # Added trust_remote_code for custom tokenizers
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token # Ensure pad token is set

    logger.info(f"Loading C4 dataset for languages: {args.c4_langs} (Streaming: {args.dataset_streaming})")

    # Load multiple language datasets
    raw_datasets_list = []
    for lang in args.c4_langs:
        try:
            ds = load_dataset("allenai/c4", lang, streaming=args.dataset_streaming, split="train", trust_remote_code=True)
            raw_datasets_list.append(ds)
        except Exception as e:
            logger.warning(f"Failed to load C4 dataset for language '{lang}': {e}")
    
    if not raw_datasets_list:
        raise ValueError("No datasets were successfully loaded")
    
    # Remove timestamp and url fields from each dataset to avoid type mismatches when interleaving
    processed_datasets = []
    for i, ds_part in enumerate(raw_datasets_list): # Changed variable name for clarity
        if ds_part is not None and hasattr(ds_part, 'column_names') and ds_part.column_names is not None:
            logger.info(f"Processing dataset part {i}: Original columns: {ds_part.column_names}")
            columns_to_remove = [col for col in ["timestamp", "url"] if col in ds_part.column_names]
            if columns_to_remove:
                logger.info(f"Dataset part {i}: Attempting to remove columns: {columns_to_remove}")
                ds_part = ds_part.remove_columns(columns_to_remove)
                logger.info(f"Dataset part {i}: Columns after removal: {ds_part.column_names}")
            else:
                logger.info(f"Dataset part {i}: No 'timestamp' or 'url' columns to remove.")
        elif ds_part is None:
            logger.warning(f"Dataset part {i} is None.")
        else:
            logger.warning(f"Dataset part {i} has no column_names attribute or it's None. Type: {type(ds_part)}")
        processed_datasets.append(ds_part)
    
    raw_datasets_list = processed_datasets
    
    raw_dataset = interleave_datasets(raw_datasets_list)

    if args.max_input_lines_total is not None:
        logger.info(f"Limiting total input lines to process to: {args.max_input_lines_total}")
        raw_dataset = raw_dataset.take(args.max_input_lines_total)
        logger.info(f"raw_dataset after take({args.max_input_lines_total}): {raw_dataset}")
        # Try to log the first item to see if data is actually being read
        try:
            first_item = next(iter(raw_dataset))
            logger.info(f"First item from raw_dataset (after take): {first_item}")
            # Re-construct raw_dataset as iter can consume the first item if not careful with how dataset is used later
            # This is a common pattern if you need to peek and then reuse.
            # For this script, raw_dataset is used in a .map() operation,
            # which will iterate over it. Peeking here might be fine if the dataset can be iterated multiple times
            # or if it's re-fetched. For HF streaming datasets, .map iterates once.
            # A safer way for streaming is to map a logging function or rely on later logs.
            # For now, let's just log the object itself, and if it's iterable, the next log points will tell us more.
        except StopIteration:
            logger.warning("raw_dataset is empty after take().")
        except Exception as e:
            logger.warning(f"Could not retrieve first item from raw_dataset after take(): {e}")


    def tokenize_function(examples, **kwargs):
        texts = examples["text"]
        logger.info(f"tokenize_function received texts: {texts}") # Changed to info
        # Filter out None or empty strings if any slip through, though map might handle some.
        valid_texts = [t for t in texts if t and isinstance(t, str) and t.strip()]
        if not valid_texts:
            logger.warning("tokenize_function: valid_texts is empty.")
            return {"input_ids": [], "attention_mask": []}
        else:
            logger.info(f"tokenize_function: number of valid_texts: {len(valid_texts)}") # Changed to info
        # Tokenize. `truncation=False` because grouping will handle fixed lengths.
        output = tokenizer(valid_texts, truncation=False)
        logger.info(f"tokenize_function output: {output}") # Changed to info
        return output

    logger.info(f"Mapping tokenization function over the dataset using {args.num_proc} processes...")
    # Batched=True is important for efficiency. Batch size for map can be default or tuned.
    # Note: num_proc doesn't work with streaming datasets, so we'll only use it for non-streaming

    # Bind the tokenizer object to the tokenize_function
    bound_tokenize_function = partial(tokenize_function, tokenizer_obj=tokenizer)

    if args.dataset_streaming:
        tokenized_dataset = raw_dataset.map(
            bound_tokenize_function, # Use the bound function
            batched=True,
            remove_columns=["text", "c4_language"] # Remove original and any lang-specific columns
        )
    else:
        tokenized_dataset = raw_dataset.map(
            bound_tokenize_function, # Use the bound function
            batched=True,
            num_proc=args.num_proc,
            remove_columns=["text", "c4_language"] # Remove original and any lang-specific columns
        )

    # Grouping into Blocks
    def group_texts(examples):
        # Define expected keys for processing
        expected_keys = ['input_ids', 'attention_mask']
        concatenated_examples = {}
        for k in expected_keys:
            if k in examples and examples[k] is not None:
                # Ensure all items intended for sum are lists (e.g. list of token_ids)
                # and filter out any None items within those lists of lists.
                processed_list = [item for item in examples[k] if item is not None and isinstance(item, list)]
                if not processed_list and examples[k]: # If examples[k] was not empty but processed_list is, it means it contained non-list or all None items
                     logger.warning(f"group_texts: Key '{k}' contained non-list items or only None items after filtering. Original items count: {len(examples[k])}")
                concatenated_examples[k] = sum(processed_list, [])
            elif k in examples and examples[k] is None:
                logger.warning(f"group_texts: Key '{k}' was present but its value was None. Skipping concatenation for this key.")
                concatenated_examples[k] = [] # Initialize as empty list if key was None
            elif k not in examples:
                logger.warning(f"group_texts: Expected key '{k}' not found in examples. Skipping concatenation for this key.")
                concatenated_examples[k] = [] # Initialize as empty list if key missing

        logger.debug(f"group_texts: keys in concatenated_examples: {list(concatenated_examples.keys())}") # Kept as debug

        input_ids_list = concatenated_examples.get('input_ids', [])
        total_length = len(input_ids_list)
        logger.info(f"group_texts: total_length of concatenated input_ids: {total_length}, args.block_size: {args.block_size}")

        if total_length == 0:
            logger.warning("group_texts: total_length is 0, returning empty.")
            return {k: [] for k in examples.keys()} # Return empty if no data

        # We drop the small remainder.
        if total_length >= args.block_size:
            total_length = (total_length // args.block_size) * args.block_size
        else: # If total length is less than block_size, effectively no full blocks
            logger.warning(f"group_texts: total_length {total_length} is less than block_size {args.block_size}, returning empty.")
            return {k: [] for k in examples.keys()} # Return empty

        result = {
            k: [t[i : i + args.block_size] for i in range(0, total_length, args.block_size)]
            for k, t in concatenated_examples.items()
        }
        logger.info(f"group_texts: result being returned: {{key: len(v) for key, v in result.items() if isinstance(v, list)}}") # Changed to info
        return result

    logger.info(f"Mapping grouping function (block_size={args.block_size}) over the tokenized dataset using {args.num_proc} processes...")
    # Batched=True for group_texts as well.
    # The batch size for this map can also be tuned if needed, default is 1000.
    # Filter out empty results from tokenize_function or group_texts before attempting to save.
    if args.dataset_streaming:
        lm_dataset_before_filter = tokenized_dataset.map(group_texts, batched=True)
        lm_dataset = lm_dataset_before_filter.filter(lambda x: len(x['input_ids']) > 0)
    else:
        lm_dataset_before_filter = tokenized_dataset.map(group_texts, batched=True, num_proc=args.num_proc)
        lm_dataset = lm_dataset_before_filter.filter(lambda x: len(x['input_ids']) > 0, num_proc=args.num_proc)

    logger.info(f"lm_dataset object after filter: {lm_dataset}")
    try:
        preview_dataset = lm_dataset.take(5) # take first 5 for preview
        preview_samples = list(preview_dataset) # materialize them
        if preview_samples:
            logger.info(f"Preview of lm_dataset (first {len(preview_samples)} samples):")
            for i, sample in enumerate(preview_samples):
                 logger.info(f"lm_dataset preview sample {i}: {{key: type(v) for key,v in sample.items()}} input_ids_len: {len(sample.get('input_ids', []))}")
        else:
            logger.warning("lm_dataset preview (first 5 samples) is empty.")
        # lm_dataset itself is NOT consumed here by .take() for the main processing loop,
        # as .take() on an IterableDataset usually returns a new iterable.
    except Exception as e:
        logger.error(f"Error when trying to generate preview from lm_dataset: {e}", exc_info=True)

    os.makedirs(args.output_path, exist_ok=True)
    logger.info(f"Output directory set to: {args.output_path}")

    # Iterate, save to Parquet in shards
    shard_index = 0
    processed_samples_in_current_shard = 0
    current_shard_data = {k: [] for k in ["input_ids", "attention_mask"]}

    logger.info(f"Starting processing and sharding. Max samples per shard: {args.max_samples_per_shard}")

    # Create progress bar for sharding
    pbar = tqdm(desc="Processing samples", unit="samples")
    
    for processed_example in lm_dataset: # Iterate through the final processed dataset
        logger.info(f"Processing loop: received processed_example. Keys: {list(processed_example.keys())}, input_ids length: {len(processed_example.get('input_ids', []))}")
        if not processed_example['input_ids']: # Should be filtered out by .filter, but as a safeguard
            logger.warning("Processing loop: Encountered an example with no input_ids AFTER filter. Skipping.")
            continue

        current_shard_data["input_ids"].append(processed_example["input_ids"])
        current_shard_data["attention_mask"].append(processed_example["attention_mask"])
        processed_samples_in_current_shard += 1
        pbar.update(1)

        if processed_samples_in_current_shard >= args.max_samples_per_shard:
            shard_file_path = os.path.join(args.output_path, f"shard_{shard_index}.parquet")
            logger.info(f"Attempting to write shard {shard_index} with {processed_samples_in_current_shard} samples. Data keys: {list(current_shard_data.keys())}")
            try:
                table = pa.Table.from_pydict(current_shard_data)
                pq.write_table(table, shard_file_path)
                logger.info(f"Saved shard {shard_index} to {shard_file_path} with {processed_samples_in_current_shard} samples.")
            except Exception as e:
                logger.error(f"Error writing shard {shard_index} to {shard_file_path}: {e}")

            shard_index += 1
            processed_samples_in_current_shard = 0
            current_shard_data = {k: [] for k in ["input_ids", "attention_mask"]}

    # Save any remaining data in the last shard
    logger.info(f"Final check: processed_samples_in_current_shard = {processed_samples_in_current_shard}")
    if processed_samples_in_current_shard > 0:
        shard_file_path = os.path.join(args.output_path, f"shard_{shard_index}.parquet")
        logger.info(f"Attempting to write FINAL shard {shard_index} with {processed_samples_in_current_shard} samples. Data keys: {list(current_shard_data.keys())}")
        try:
            table = pa.Table.from_pydict(current_shard_data)
            pq.write_table(table, shard_file_path)
            logger.info(f"Saved final shard {shard_index} to {shard_file_path} with {processed_samples_in_current_shard} samples.")
        except Exception as e:
            logger.error(f"Error writing final shard {shard_index} to {shard_file_path}: {e}")

    pbar.close()
    logger.info(f"Reached end of main processing. Final shard_index: {shard_index}, final processed_samples_in_current_shard: {processed_samples_in_current_shard}")
    logger.info("Dataset pre-tokenization and sharding complete.")
    if shard_index == 0 and processed_samples_in_current_shard == 0:
        logger.warning("No data was processed or saved. Check dataset source or processing parameters (e.g. max_input_lines_total, block_size).")
    logger.info("End of main function reached.")

if __name__ == "__main__":
    main()
