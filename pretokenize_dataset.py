import argparse
import os
import gc
import psutil
import datasets
from datasets import load_dataset, interleave_datasets, concatenate_datasets
from functools import partial # Add this import
from transformers import AutoTokenizer
from transformers.utils import logging
import pyarrow.parquet as pq
import pyarrow as pa
from tqdm import tqdm
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
import itertools

logger = logging.get_logger(__name__)
# datasets.disable_caching() # Commented out - this prevents using cached datasets!

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
    parser.set_defaults(dataset_streaming=False)  # Changed default to False for better performance
    parser.add_argument(
        "--c4_langs",
        type=str,
        nargs='+',
        default=["en", "ja", "ko", "zh"],
        help="List of C4 languages to process."
    )
    parser.add_argument(
        "--language",
        type=str,
        default=None,
        help="Process only a single language. Options: en, ja, ko, cn (simplified Chinese), tw (traditional Chinese)"
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
        help="Number of processes to use for parallel tokenization and grouping (only effective if --no_dataset_streaming is used). Default: CPU count"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=None,
        help="Number of worker threads for data loading. If not specified, uses num_proc value."
    )
    parser.add_argument(
        "--save_every_n_lines",
        type=int,
        default=None,
        help="Save checkpoint every N lines processed. If specified, enables resumable processing."
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from previous checkpoint if available."
    )
    parser.add_argument(
        "--interleave_by_document",
        action="store_true",
        default=True,
        help="Interleave complete documents from different languages instead of sentences."
    )
    parser.add_argument(
        "--documents_per_batch",
        type=int,
        default=1000,
        help="Number of documents to process per batch for interleaving."
    )
    parser.add_argument(
        "--process_languages_separately",
        action="store_true",
        default=False,
        help="Process each language dataset separately to reduce memory usage."
    )
    parser.add_argument(
        "--max_memory_gb",
        type=float,
        default=8.0,
        help="Maximum memory usage in GB before forcing garbage collection."
    )
    parser.add_argument(
        "--memory_check_interval",
        type=int,
        default=1000,
        help="Check memory usage every N documents."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1000,
        help="Batch size for map operations. Larger values can improve throughput."
    )
    parser.add_argument(
        "--writer_batch_size",
        type=int,
        default=1000,
        help="Batch size for writing parquet files."
    )
    parser.add_argument(
        "--save_batch_size",
        type=int,
        default=100000,
        help="Save partial shard every N samples for incremental saving. Set to 0 to disable."
    )


    args = parser.parse_args()
    
    # If single language specified, override c4_langs
    if args.language:
        args.c4_langs = [args.language]
        logger.info(f"Processing single language: {args.language}")
    
    # Set num_proc to use all CPU cores by default
    if args.num_proc is None:
        args.num_proc = multiprocessing.cpu_count()
    
    # Set num_workers if not specified
    if args.num_workers is None:
        args.num_workers = min(4, args.num_proc)  # Limit workers for I/O operations

    logging.set_verbosity_info()

    # Helper function to get memory usage
    def get_memory_usage():
        """Get current memory usage in GB."""
        return psutil.Process().memory_info().rss / (1024 ** 3)
    
    def log_memory(stage):
        """Log current memory usage."""
        mem_gb = get_memory_usage()
        logger.info(f"[{stage}] Memory usage: {mem_gb:.2f} GB")

    logger.info(f"Loading tokenizer from: {args.tokenizer_path}")
    log_memory("Start")
    # Use AutoTokenizer which can load ModernGPT2TokenizerFast or other tokenizer types
    # This assumes tokenizer_path is a directory containing tokenizer.json
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True) # Added trust_remote_code for custom tokenizers
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token # Ensure pad token is set

    logger.info(f"Loading C4 dataset for languages: {args.c4_langs} (Streaming: {args.dataset_streaming})")
    log_memory("After tokenizer load")

    # Check if we should process languages separately
    if args.process_languages_separately:
        logger.info("Processing languages separately to reduce memory usage")
        process_languages_separately(args, tokenizer, get_memory_usage, log_memory)
        return

    # Load multiple language datasets in parallel
    def load_single_dataset(lang):
        try:
            # Use C4 for English, FineWeb2 for CJK languages
            if lang == "en":
                ds = load_dataset("c4", lang, streaming=args.dataset_streaming, split="train", trust_remote_code=True)
            elif lang == "ja":
                ds = load_dataset("HuggingFaceFW/fineweb-2", "jpn_Jpan", streaming=args.dataset_streaming, split="train", trust_remote_code=True)
            elif lang == "ko":
                ds = load_dataset("HuggingFaceFW/fineweb-2", "kor_Hang", streaming=args.dataset_streaming, split="train", trust_remote_code=True)
            elif lang == "cn":
                # Simplified Chinese (Mandarin)
                ds = load_dataset("HuggingFaceFW/fineweb-2", "cmn_Hani", streaming=args.dataset_streaming, split="train", trust_remote_code=True)
            elif lang == "tw":
                # Traditional Chinese (Cantonese)
                ds = load_dataset("HuggingFaceFW/fineweb-2", "yue_Hani", streaming=args.dataset_streaming, split="train", trust_remote_code=True)
            elif lang == "zh":
                # For backward compatibility, default to simplified Chinese
                logger.warning("Language code 'zh' is deprecated. Use 'cn' for simplified Chinese or 'tw' for traditional Chinese. Defaulting to 'cn'.")
                ds = load_dataset("HuggingFaceFW/fineweb-2", "cmn_Hani", streaming=args.dataset_streaming, split="train", trust_remote_code=True)
            else:
                raise ValueError(f"Unsupported language: {lang}")
            return (lang, ds)
        except Exception as e:
            logger.warning(f"Failed to load dataset for language '{lang}': {e}")
            return (lang, None)
    
    # Use ThreadPoolExecutor for parallel dataset loading
    with ThreadPoolExecutor(max_workers=len(args.c4_langs)) as executor:
        dataset_results = list(executor.map(load_single_dataset, args.c4_langs))
    
    raw_datasets_list = []
    for lang, ds in dataset_results:
        if ds is not None:
            raw_datasets_list.append(ds)
            logger.info(f"Successfully loaded dataset for language: {lang}")
            log_memory(f"After loading {lang}")
    
    if not raw_datasets_list:
        raise ValueError("No datasets were successfully loaded")
    
    # Remove timestamp and url fields from each dataset to avoid type mismatches when interleaving
    processed_datasets = []
    for i, ds_part in enumerate(raw_datasets_list): # Changed variable name for clarity
        if ds_part is not None and hasattr(ds_part, 'column_names') and ds_part.column_names is not None:
            logger.info(f"Processing dataset part {i}: Original columns: {ds_part.column_names}")
            # C4 has timestamp and url, FineWeb2 has different columns
            # Try to detect and remove non-text columns dynamically
            columns_to_keep = ['text']
            columns_to_remove = [col for col in ds_part.column_names if col not in columns_to_keep]
            if columns_to_remove:
                logger.info(f"Dataset part {i}: Attempting to remove columns: {columns_to_remove}")
                ds_part = ds_part.remove_columns(columns_to_remove)
                logger.info(f"Dataset part {i}: Columns after removal: {ds_part.column_names}")
            else:
                logger.info(f"Dataset part {i}: No columns to remove.")
        elif ds_part is None:
            logger.warning(f"Dataset part {i} is None.")
        else:
            logger.warning(f"Dataset part {i} has no column_names attribute or it's None. Type: {type(ds_part)}")
        processed_datasets.append(ds_part)
    
    raw_datasets_list = processed_datasets
    
    # If interleaving by document, we need a custom approach
    if args.interleave_by_document:
        logger.info("Interleaving datasets by complete documents...")
        
        class DocumentInterleavedGenerator:
            def __init__(self, datasets, c4_langs):
                self.datasets = datasets
                self.c4_langs = c4_langs
                
            def __call__(self):
                """Generator that yields complete documents from each language in round-robin fashion."""
                # Create iterators for each dataset
                dataset_iterators = [iter(ds) for ds in self.datasets]
                dataset_exhausted = [False] * len(dataset_iterators)
                documents_yielded = 0
                
                while not all(dataset_exhausted):
                    for i, iterator in enumerate(dataset_iterators):
                        if dataset_exhausted[i]:
                            continue
                        try:
                            # Get next document from this language
                            document = next(iterator)
                            documents_yielded += 1
                            yield document
                            
                            # Yield checkpoint info periodically
                            if documents_yielded % 10000 == 0:
                                logger.info(f"Processed {documents_yielded} documents so far...")
                        except StopIteration:
                            dataset_exhausted[i] = True
                            logger.info(f"Dataset {i} (language: {self.c4_langs[i] if i < len(self.c4_langs) else 'unknown'}) exhausted.")
        
        # Create a new dataset from the generator
        from datasets import Dataset, IterableDataset
        if args.dataset_streaming:
            generator_instance = DocumentInterleavedGenerator(raw_datasets_list, args.c4_langs)
            raw_dataset = IterableDataset.from_generator(generator_instance)
        else:
            # For non-streaming, we need to collect all documents first
            generator_instance = DocumentInterleavedGenerator(raw_datasets_list, args.c4_langs)
            all_documents = list(generator_instance())
            raw_dataset = Dataset.from_list(all_documents)
    else:
        # Original sentence-level interleaving
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


    def tokenize_function(examples, tokenizer_obj):
        texts = examples["text"]
        
        # Process complete documents instead of splitting by sentence
        if args.interleave_by_document:
            # Keep documents intact - don't split them
            valid_texts = []
            for text in texts:
                if text and isinstance(text, str) and text.strip():
                    # Keep the full document text without splitting
                    valid_texts.append(text)
            
            if not valid_texts:
                logger.warning("tokenize_function: valid_texts is empty.")
                return {"input_ids": [], "attention_mask": []}
            
            logger.debug(f"tokenize_function: processing {len(valid_texts)} complete documents")
            
            # Tokenize complete documents
            # Truncate to block_size to avoid sequences longer than model can handle
            output = tokenizer_obj(valid_texts, truncation=True, max_length=args.block_size, padding=False)
            
            # Add document boundaries for later processing
            # We'll add a special token between documents if needed
            if hasattr(tokenizer_obj, 'eos_token_id') and tokenizer_obj.eos_token_id is not None:
                for i in range(len(output['input_ids'])):
                    # Ensure each document ends with EOS token for proper separation
                    if output['input_ids'][i] and output['input_ids'][i][-1] != tokenizer_obj.eos_token_id:
                        output['input_ids'][i].append(tokenizer_obj.eos_token_id)
                        output['attention_mask'][i].append(1)
        else:
            # Original per-sentence processing
            logger.info(f"tokenize_function received texts: {texts}")
            valid_texts = [t for t in texts if t and isinstance(t, str) and t.strip()]
            if not valid_texts:
                logger.warning("tokenize_function: valid_texts is empty.")
                return {"input_ids": [], "attention_mask": []}
            else:
                logger.info(f"tokenize_function: number of valid_texts: {len(valid_texts)}")
            output = tokenizer_obj(valid_texts, truncation=True, max_length=args.block_size)
            logger.info(f"tokenize_function output: {output}")
        
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
            batch_size=args.batch_size,
            remove_columns=["text"] # Remove original text column
        )
    else:
        tokenized_dataset = raw_dataset.map(
            bound_tokenize_function, # Use the bound function
            batched=True,
            batch_size=args.batch_size,
            num_proc=args.num_proc,
            remove_columns=["text"] # Remove original text column
        )

    # Grouping into Blocks
    def group_texts(examples):
        # Define expected keys for processing
        expected_keys = ['input_ids', 'attention_mask']
        concatenated_examples = {}
        
        if args.interleave_by_document:
            # For document-level processing, we want to preserve document boundaries
            # while still creating fixed-size blocks
            for k in expected_keys:
                if k in examples and examples[k] is not None:
                    processed_list = [item for item in examples[k] if item is not None and isinstance(item, list)]
                    if not processed_list and examples[k]:
                        logger.warning(f"group_texts: Key '{k}' contained non-list items or only None items after filtering.")
                    # Concatenate with document boundaries preserved
                    # Use itertools.chain for efficient concatenation
                    concatenated_examples[k] = list(itertools.chain.from_iterable(processed_list))
                else:
                    concatenated_examples[k] = []
            
            # Create blocks but try to avoid splitting in the middle of documents
            # This is a simple approach - more sophisticated approaches could track document boundaries
            input_ids_list = concatenated_examples.get('input_ids', [])
            total_length = len(input_ids_list)
            
            if total_length == 0:
                logger.warning("group_texts: total_length is 0, returning empty.")
                return {k: [] for k in expected_keys}
            
            # Create blocks of the specified size
            # Note: This may split documents, but maintains better training efficiency
            # For truly document-aligned training, you'd need a different approach
            if total_length >= args.block_size:
                total_length = (total_length // args.block_size) * args.block_size
            else:
                logger.warning(f"group_texts: total_length {total_length} is less than block_size {args.block_size}, returning empty.")
                return {k: [] for k in expected_keys}
            
            result = {
                k: [t[i : i + args.block_size] for i in range(0, total_length, args.block_size)]
                for k, t in concatenated_examples.items()
            }
            
            logger.debug(f"group_texts: Created {len(result.get('input_ids', []))} blocks of size {args.block_size}")
        else:
            # Original sentence-level processing
            for k in expected_keys:
                if k in examples and examples[k] is not None:
                    processed_list = [item for item in examples[k] if item is not None and isinstance(item, list)]
                    if not processed_list and examples[k]:
                         logger.warning(f"group_texts: Key '{k}' contained non-list items or only None items after filtering. Original items count: {len(examples[k])}")
                    # Use itertools.chain for efficient concatenation
                    concatenated_examples[k] = list(itertools.chain.from_iterable(processed_list))
                elif k in examples and examples[k] is None:
                    logger.warning(f"group_texts: Key '{k}' was present but its value was None. Skipping concatenation for this key.")
                    concatenated_examples[k] = []
                elif k not in examples:
                    logger.warning(f"group_texts: Expected key '{k}' not found in examples. Skipping concatenation for this key.")
                    concatenated_examples[k] = []

            logger.debug(f"group_texts: keys in concatenated_examples: {list(concatenated_examples.keys())}")

            input_ids_list = concatenated_examples.get('input_ids', [])
            total_length = len(input_ids_list)
            logger.info(f"group_texts: total_length of concatenated input_ids: {total_length}, args.block_size: {args.block_size}")

            if total_length == 0:
                logger.warning("group_texts: total_length is 0, returning empty.")
                return {k: [] for k in examples.keys()}

            if total_length >= args.block_size:
                total_length = (total_length // args.block_size) * args.block_size
            else:
                logger.warning(f"group_texts: total_length {total_length} is less than block_size {args.block_size}, returning empty.")
                return {k: [] for k in examples.keys()}

            result = {
                k: [t[i : i + args.block_size] for i in range(0, total_length, args.block_size)]
                for k, t in concatenated_examples.items()
            }
            logger.info(f"group_texts: result being returned: {{key: len(v) for key, v in result.items() if isinstance(v, list)}}")
        
        return result

    logger.info(f"Mapping grouping function (block_size={args.block_size}) over the tokenized dataset using {args.num_proc} processes...")
    # Batched=True for group_texts as well.
    # The batch size for this map can also be tuned if needed, default is 1000.
    # Filter out empty results from tokenize_function or group_texts before attempting to save.
    if args.dataset_streaming:
        lm_dataset_before_filter = tokenized_dataset.map(
            group_texts, 
            batched=True,
            batch_size=args.batch_size
        )
        lm_dataset = lm_dataset_before_filter.filter(lambda x: len(x['input_ids']) > 0)
    else:
        lm_dataset_before_filter = tokenized_dataset.map(
            group_texts, 
            batched=True, 
            batch_size=args.batch_size,
            num_proc=args.num_proc
        )
        lm_dataset = lm_dataset_before_filter.filter(
            lambda x: len(x['input_ids']) > 0, 
            num_proc=args.num_proc,
            batch_size=args.batch_size
        )

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

    # Load checkpoint if resuming
    checkpoint_path = os.path.join(args.output_path, "checkpoint.json")
    total_lines_processed = 0
    shard_index = 0
    samples_in_current_shard = 0
    
    if args.resume and os.path.exists(checkpoint_path):
        import json
        with open(checkpoint_path, 'r') as f:
            checkpoint = json.load(f)
        total_lines_processed = checkpoint.get('total_lines_processed', 0)
        shard_index = checkpoint.get('current_shard_index', 0)
        samples_in_current_shard = checkpoint.get('samples_in_current_shard', 0)
        
        # Check if we have a partial shard
        shard_file_path = os.path.join(args.output_path, f"shard_{shard_index}.parquet")
        if os.path.exists(shard_file_path):
            existing_table = pq.read_table(shard_file_path)
            actual_samples = existing_table.num_rows
            if actual_samples != samples_in_current_shard:
                logger.warning(f"Checkpoint mismatch: expected {samples_in_current_shard} samples in shard {shard_index}, found {actual_samples}")
                samples_in_current_shard = actual_samples
        
        logger.info(f"Resuming from checkpoint: {total_lines_processed} lines processed, shard {shard_index} with {samples_in_current_shard} samples")
        
        # Skip already processed lines
        if total_lines_processed > 0:
            logger.info(f"Skipping first {total_lines_processed} lines...")
            skip_pbar = tqdm(total=total_lines_processed, desc="Skipping processed lines")
            for _ in range(total_lines_processed):
                try:
                    next(iter(lm_dataset))
                    skip_pbar.update(1)
                except StopIteration:
                    break
            skip_pbar.close()

    # Optimized batch processing for Parquet writing
    def write_shard(shard_data, shard_idx, append=False):
        shard_file_path = os.path.join(args.output_path, f"shard_{shard_idx}.parquet")
        try:
            table = pa.Table.from_pydict(shard_data)
            if append and os.path.exists(shard_file_path):
                # Read existing data and append
                existing_table = pq.read_table(shard_file_path)
                combined_table = pa.concat_tables([existing_table, table])
                pq.write_table(combined_table, shard_file_path, compression='snappy')
                logger.info(f"Appended to shard {shard_idx} at {shard_file_path} with {len(shard_data['input_ids'])} new samples (total: {combined_table.num_rows}).")
            else:
                pq.write_table(table, shard_file_path, compression='snappy')
                logger.info(f"Saved shard {shard_idx} to {shard_file_path} with {len(shard_data['input_ids'])} samples.")
            return True
        except Exception as e:
            logger.error(f"Error writing shard {shard_idx} to {shard_file_path}: {e}")
            return False

    # Process dataset in batches
    processed_samples_in_current_shard = samples_in_current_shard  # Start from checkpoint
    current_shard_data = {k: [] for k in ["input_ids", "attention_mask"]}
    lines_since_checkpoint = 0
    samples_since_last_save = 0
    batch_buffer = []

    logger.info(f"Starting processing and sharding. Max samples per shard: {args.max_samples_per_shard}")
    if args.save_batch_size > 0:
        logger.info(f"Incremental save enabled: saving every {args.save_batch_size} samples")

    # Create progress bar for sharding
    pbar = tqdm(desc="Processing samples", unit="samples", initial=total_lines_processed)
    
    # Process in batches for better performance
    for processed_example in lm_dataset:
        if not processed_example['input_ids']:
            logger.warning("Processing loop: Encountered an example with no input_ids AFTER filter. Skipping.")
            continue

        current_shard_data["input_ids"].append(processed_example["input_ids"])
        current_shard_data["attention_mask"].append(processed_example["attention_mask"])
        processed_samples_in_current_shard += 1
        total_lines_processed += 1
        lines_since_checkpoint += 1
        samples_since_last_save += 1
        pbar.update(1)
        
        # Save partial shard if we've accumulated enough samples
        if args.save_batch_size > 0 and samples_since_last_save >= args.save_batch_size:
            # Write current batch to shard (append mode)
            write_shard(current_shard_data, shard_index, append=True)
            current_shard_data = {k: [] for k in ["input_ids", "attention_mask"]}
            samples_since_last_save = 0
            
            # Update checkpoint
            import json
            checkpoint_data = {
                'total_lines_processed': total_lines_processed,
                'current_shard_index': shard_index,
                'samples_in_current_shard': processed_samples_in_current_shard
            }
            with open(checkpoint_path, 'w') as f:
                json.dump(checkpoint_data, f)
        
        # Save checkpoint if needed (line-based checkpoint)
        if args.save_every_n_lines and lines_since_checkpoint >= args.save_every_n_lines:
            # Save any pending data first
            if current_shard_data["input_ids"]:
                write_shard(current_shard_data, shard_index, append=True)
                current_shard_data = {k: [] for k in ["input_ids", "attention_mask"]}
                samples_since_last_save = 0
            
            import json
            checkpoint_data = {
                'total_lines_processed': total_lines_processed,
                'current_shard_index': shard_index,
                'samples_in_current_shard': processed_samples_in_current_shard
            }
            with open(checkpoint_path, 'w') as f:
                json.dump(checkpoint_data, f)
            logger.info(f"Checkpoint saved: {total_lines_processed} lines processed")
            lines_since_checkpoint = 0

        if processed_samples_in_current_shard >= args.max_samples_per_shard:
            # Save any remaining data in buffer
            if current_shard_data["input_ids"]:
                write_shard(current_shard_data, shard_index, append=True)
                current_shard_data = {k: [] for k in ["input_ids", "attention_mask"]}
                samples_since_last_save = 0
            
            # Move to next shard
            shard_index += 1
            processed_samples_in_current_shard = 0

    # Save any remaining data in the last shard
    logger.info(f"Final check: processed_samples_in_current_shard = {processed_samples_in_current_shard}")
    if current_shard_data["input_ids"]:
        write_shard(current_shard_data, shard_index, append=True)

    pbar.close()
    
    # Save final checkpoint
    import json
    checkpoint_data = {
        'total_lines_processed': total_lines_processed,
        'current_shard_index': shard_index,
        'samples_in_current_shard': processed_samples_in_current_shard
    }
    with open(checkpoint_path, 'w') as f:
        json.dump(checkpoint_data, f)
    logger.info(f"Final checkpoint saved: {total_lines_processed} total lines processed")
    
    logger.info(f"Reached end of main processing. Final shard_index: {shard_index}, final processed_samples_in_current_shard: {processed_samples_in_current_shard}")
    logger.info(f"Total lines processed: {total_lines_processed}")
    logger.info("Dataset pre-tokenization and sharding complete.")
    if shard_index == 0 and processed_samples_in_current_shard == 0:
        logger.warning("No data was processed or saved. Check dataset source or processing parameters (e.g. max_input_lines_total, block_size).")
    logger.info("End of main function reached.")


def process_languages_separately(args, tokenizer, get_memory_usage, log_memory):
    """Process each language dataset separately to minimize memory usage."""
    import json
    
    os.makedirs(args.output_path, exist_ok=True)
    
    # Write dataset info file for easy loading
    dataset_info_path = os.path.join(args.output_path, "_dataset_info.json")
    dataset_info = {
        "format": "parquet",
        "splits": {"train": {"name": "train", "num_examples": "in_progress"}},
        "description": "Pre-tokenized C4 dataset",
        "features": {
            "input_ids": {"dtype": "int64", "shape": [args.block_size]},
            "attention_mask": {"dtype": "int64", "shape": [args.block_size]}
        }
    }
    with open(dataset_info_path, 'w') as f:
        json.dump(dataset_info, f, indent=2)
    
    # Global tracking across all languages
    global_shard_index = 0
    global_total_lines = 0
    
    # Load checkpoint if resuming
    checkpoint_path = os.path.join(args.output_path, "checkpoint.json")
    current_language = None
    current_language_lines = 0
    samples_in_current_shard = 0
    if args.resume and os.path.exists(checkpoint_path):
        with open(checkpoint_path, 'r') as f:
            checkpoint = json.load(f)
        global_shard_index = checkpoint.get('last_shard_index', 0)
        if global_shard_index is None:
            global_shard_index = 0
        global_total_lines = checkpoint.get('total_lines_processed', 0)
        processed_languages = checkpoint.get('processed_languages', [])
        current_language = checkpoint.get('current_language', None)
        current_language_lines = checkpoint.get('current_language_lines', 0)
        samples_in_current_shard = checkpoint.get('samples_in_current_shard', 0)
        
        # Check if we have a partial shard
        shard_file_path = os.path.join(args.output_path, f"shard_{global_shard_index}.parquet")
        if os.path.exists(shard_file_path):
            existing_table = pq.read_table(shard_file_path)
            actual_samples = existing_table.num_rows
            if actual_samples != samples_in_current_shard:
                logger.warning(f"Checkpoint mismatch: expected {samples_in_current_shard} samples in shard {global_shard_index}, found {actual_samples}")
                samples_in_current_shard = actual_samples
        
        logger.info(f"Resuming from checkpoint: {global_total_lines} lines processed, {len(processed_languages)} languages completed")
        if current_language:
            logger.info(f"Resuming {current_language} from line {current_language_lines}")
        logger.info(f"Current shard: {global_shard_index} with {samples_in_current_shard} samples")
    else:
        processed_languages = []
    
    # Process each language separately
    for lang in args.c4_langs:
        if lang in processed_languages and lang != current_language:
            logger.info(f"Skipping already processed language: {lang}")
            continue
            
        logger.info(f"\n{'='*50}")
        logger.info(f"Processing language: {lang}")
        logger.info(f"{'='*50}")
        
        try:
            # Load single language dataset
            logger.info(f"Loading dataset for language: {lang} (streaming={args.dataset_streaming})...")
            logger.info(f"This may take a while if streaming is disabled and dataset needs to be downloaded/cached...")
            
            # Use C4 for English, FineWeb2 for CJK languages
            if lang == "en":
                raw_dataset = load_dataset("c4", lang, streaming=args.dataset_streaming, split="train", trust_remote_code=True)
            elif lang == "ja":
                raw_dataset = load_dataset("HuggingFaceFW/fineweb-2", "jpn_Jpan", streaming=args.dataset_streaming, split="train", trust_remote_code=True)
            elif lang == "ko":
                raw_dataset = load_dataset("HuggingFaceFW/fineweb-2", "kor_Hang", streaming=args.dataset_streaming, split="train", trust_remote_code=True)
            elif lang == "cn":
                # Simplified Chinese (Mandarin)
                raw_dataset = load_dataset("HuggingFaceFW/fineweb-2", "cmn_Hani", streaming=args.dataset_streaming, split="train", trust_remote_code=True)
            elif lang == "tw":
                # Traditional Chinese (Cantonese)
                raw_dataset = load_dataset("HuggingFaceFW/fineweb-2", "yue_Hani", streaming=args.dataset_streaming, split="train", trust_remote_code=True)
            elif lang == "zh":
                # For backward compatibility
                logger.warning("Language code 'zh' is deprecated. Use 'cn' for simplified Chinese or 'tw' for traditional Chinese. Defaulting to 'cn'.")
                raw_dataset = load_dataset("HuggingFaceFW/fineweb-2", "cmn_Hani", streaming=args.dataset_streaming, split="train", trust_remote_code=True)
            else:
                raise ValueError(f"Unsupported language: {lang}")
            
            logger.info(f"Successfully loaded dataset for {lang}")
            # Check if using cache (different cache paths for C4 vs FineWeb2)
            if lang == "en":
                cache_dir = os.path.expanduser(f"~/.cache/huggingface/datasets/allenai___c4/{lang}/0.0.0/")
            else:
                # FineWeb2 cache path is different
                cache_dir = os.path.expanduser(f"~/.cache/huggingface/datasets/HuggingFaceFW___fineweb-2/")
            if os.path.exists(cache_dir) and not args.dataset_streaming:
                logger.info(f"Using cached dataset from: {cache_dir}")
            if hasattr(raw_dataset, '__len__'):
                logger.info(f"Dataset size: {len(raw_dataset)} examples")
            else:
                logger.info(f"Dataset is in streaming mode")
            log_memory(f"After loading {lang} dataset")
            
            # Remove unnecessary columns (keep only 'text')
            if hasattr(raw_dataset, 'column_names') and raw_dataset.column_names is not None:
                columns_to_keep = ['text']
                columns_to_remove = [col for col in raw_dataset.column_names if col not in columns_to_keep]
                if columns_to_remove:
                    logger.info(f"Removing columns: {columns_to_remove}")
                    raw_dataset = raw_dataset.remove_columns(columns_to_remove)
            
            # Limit lines if specified
            if args.max_input_lines_total and global_total_lines < args.max_input_lines_total:
                remaining_lines = args.max_input_lines_total - global_total_lines
                logger.info(f"Limiting {lang} to {remaining_lines} lines")
                raw_dataset = raw_dataset.take(remaining_lines)
            
            # Tokenize function
            def tokenize_function(examples):
                texts = examples["text"]
                valid_texts = []
                for text in texts:
                    if text and isinstance(text, str) and text.strip():
                        valid_texts.append(text)
                
                if not valid_texts:
                    return {"input_ids": [], "attention_mask": []}
                
                output = tokenizer(valid_texts, truncation=True, max_length=args.block_size, padding=False)
                
                # Add EOS tokens
                if hasattr(tokenizer, 'eos_token_id') and tokenizer.eos_token_id is not None:
                    for i in range(len(output['input_ids'])):
                        if output['input_ids'][i] and output['input_ids'][i][-1] != tokenizer.eos_token_id:
                            output['input_ids'][i].append(tokenizer.eos_token_id)
                            output['attention_mask'][i].append(1)
                
                return output
            
            # Tokenize dataset
            logger.info(f"Tokenizing {lang} dataset...")
            logger.info(f"Using {'streaming' if args.dataset_streaming else f'{args.num_proc} processes'} for tokenization")
            
            if args.dataset_streaming:
                tokenized_dataset = raw_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
            else:
                # For non-streaming, the map operation shows progress automatically
                tokenized_dataset = raw_dataset.map(tokenize_function, batched=True, num_proc=args.num_proc, remove_columns=["text"])
            
            logger.info(f"Tokenization completed for {lang}")
            
            # Group into blocks
            def group_texts(examples):
                concatenated_examples = {}
                for k in ['input_ids', 'attention_mask']:
                    if k in examples and examples[k] is not None:
                        processed_list = [item for item in examples[k] if item is not None and isinstance(item, list)]
                        # Use itertools.chain for efficient concatenation
                        concatenated_examples[k] = list(itertools.chain.from_iterable(processed_list))
                    else:
                        concatenated_examples[k] = []
                
                total_length = len(concatenated_examples.get('input_ids', []))
                if total_length == 0:
                    return {k: [] for k in ['input_ids', 'attention_mask']}
                
                if total_length >= args.block_size:
                    total_length = (total_length // args.block_size) * args.block_size
                else:
                    return {k: [] for k in ['input_ids', 'attention_mask']}
                
                result = {
                    k: [t[i : i + args.block_size] for i in range(0, total_length, args.block_size)]
                    for k, t in concatenated_examples.items()
                }
                return result
            
            logger.info(f"Grouping {lang} tokens into blocks...")
            if args.dataset_streaming:
                lm_dataset = tokenized_dataset.map(group_texts, batched=True).filter(lambda x: len(x['input_ids']) > 0)
            else:
                lm_dataset = tokenized_dataset.map(group_texts, batched=True, num_proc=args.num_proc).filter(lambda x: len(x['input_ids']) > 0, num_proc=args.num_proc)
            
            # Process and save to shards
            # If resuming this language, use the checkpoint count
            if lang == current_language:
                processed_samples_in_current_shard = samples_in_current_shard
            else:
                processed_samples_in_current_shard = 0
            current_shard_data = {k: [] for k in ["input_ids", "attention_mask"]}
            lang_lines_processed = 0
            lines_since_checkpoint = 0
            samples_since_last_save = 0
            
            # Skip already processed lines if resuming this language
            skip_lines = 0
            if lang == current_language and current_language_lines > 0:
                skip_lines = current_language_lines
                logger.info(f"Skipping first {skip_lines} already processed lines in {lang}")
            
            logger.info(f"Processing {lang} samples and writing to shards...")
            if args.save_batch_size > 0:
                logger.info(f"Incremental save enabled: saving every {args.save_batch_size} samples")
            pbar = tqdm(desc=f"Processing {lang}", unit="samples", initial=skip_lines)
            
            for i, processed_example in enumerate(lm_dataset):
                # Skip already processed lines
                if i < skip_lines:
                    continue
                if not processed_example['input_ids']:
                    continue
                
                current_shard_data["input_ids"].append(processed_example["input_ids"])
                current_shard_data["attention_mask"].append(processed_example["attention_mask"])
                processed_samples_in_current_shard += 1
                lang_lines_processed += 1
                global_total_lines += 1
                lines_since_checkpoint += 1
                samples_since_last_save += 1
                pbar.update(1)
                
                # Save partial shard if we've accumulated enough samples
                if args.save_batch_size > 0 and samples_since_last_save >= args.save_batch_size:
                    # Write current batch to shard (append mode)
                    shard_file_path = os.path.join(args.output_path, f"shard_{global_shard_index}.parquet")
                    try:
                        if current_shard_data["input_ids"]:
                            table = pa.Table.from_pydict(current_shard_data)
                            if os.path.exists(shard_file_path):
                                existing_table = pq.read_table(shard_file_path)
                                combined_table = pa.concat_tables([existing_table, table])
                                pq.write_table(combined_table, shard_file_path, compression='snappy')
                                logger.info(f"Appended to shard {global_shard_index} with {len(current_shard_data['input_ids'])} new samples (total: {combined_table.num_rows})")
                            else:
                                pq.write_table(table, shard_file_path, compression='snappy')
                                logger.info(f"Created shard {global_shard_index} with {len(current_shard_data['input_ids'])} samples")
                            current_shard_data = {k: [] for k in ["input_ids", "attention_mask"]}
                            samples_since_last_save = 0
                    except Exception as e:
                        logger.error(f"Error writing partial shard: {e}")
                    
                    # Update checkpoint
                    import json
                    checkpoint_data = {
                        'total_lines_processed': global_total_lines,
                        'last_shard_index': global_shard_index,
                        'processed_languages': processed_languages,
                        'current_language': lang,
                        'current_language_lines': lang_lines_processed,
                        'samples_in_current_shard': processed_samples_in_current_shard
                    }
                    with open(checkpoint_path, 'w') as f:
                        json.dump(checkpoint_data, f)
                
                # Save checkpoint periodically (line-based)
                if args.save_every_n_lines and lines_since_checkpoint >= args.save_every_n_lines:
                    # Save any pending data first
                    if current_shard_data["input_ids"]:
                        shard_file_path = os.path.join(args.output_path, f"shard_{global_shard_index}.parquet")
                        try:
                            table = pa.Table.from_pydict(current_shard_data)
                            if os.path.exists(shard_file_path):
                                existing_table = pq.read_table(shard_file_path)
                                combined_table = pa.concat_tables([existing_table, table])
                                pq.write_table(combined_table, shard_file_path, compression='snappy')
                            else:
                                pq.write_table(table, shard_file_path, compression='snappy')
                            current_shard_data = {k: [] for k in ["input_ids", "attention_mask"]}
                            samples_since_last_save = 0
                        except Exception as e:
                            logger.error(f"Error writing partial shard at checkpoint: {e}")
                    
                    import json
                    checkpoint_data = {
                        'total_lines_processed': global_total_lines,
                        'last_shard_index': global_shard_index,
                        'processed_languages': processed_languages,
                        'current_language': lang,
                        'current_language_lines': lang_lines_processed,
                        'samples_in_current_shard': processed_samples_in_current_shard
                    }
                    with open(checkpoint_path, 'w') as f:
                        json.dump(checkpoint_data, f)
                    logger.info(f"Checkpoint saved: {global_total_lines} total lines, {lang_lines_processed} lines in {lang}")
                    lines_since_checkpoint = 0
                
                # Check memory periodically
                if lang_lines_processed % args.memory_check_interval == 0:
                    mem_usage = get_memory_usage()
                    if mem_usage > args.max_memory_gb:
                        logger.warning(f"Memory usage ({mem_usage:.2f} GB) exceeds limit. Forcing garbage collection...")
                        gc.collect()
                        log_memory(f"After GC")
                
                # Write shard when full
                if processed_samples_in_current_shard >= args.max_samples_per_shard:
                    # Save any remaining data in buffer
                    if current_shard_data["input_ids"]:
                        shard_file_path = os.path.join(args.output_path, f"shard_{global_shard_index}.parquet")
                        try:
                            table = pa.Table.from_pydict(current_shard_data)
                            if os.path.exists(shard_file_path):
                                existing_table = pq.read_table(shard_file_path)
                                combined_table = pa.concat_tables([existing_table, table])
                                pq.write_table(combined_table, shard_file_path, compression='snappy')
                            else:
                                pq.write_table(table, shard_file_path, compression='snappy')
                            current_shard_data = {k: [] for k in ["input_ids", "attention_mask"]}
                            samples_since_last_save = 0
                        except Exception as e:
                            logger.error(f"Error writing shard {global_shard_index}: {e}")
                    
                    global_shard_index += 1
                    processed_samples_in_current_shard = 0
                    gc.collect()  # Force GC after writing shard
                
                # Stop if we've reached the global limit
                if args.max_input_lines_total and global_total_lines >= args.max_input_lines_total:
                    logger.info(f"Reached global line limit of {args.max_input_lines_total}")
                    break
            
            pbar.close()
            
            # Save remaining data for this language
            if current_shard_data["input_ids"]:
                shard_file_path = os.path.join(args.output_path, f"shard_{global_shard_index}.parquet")
                try:
                    table = pa.Table.from_pydict(current_shard_data)
                    if os.path.exists(shard_file_path):
                        existing_table = pq.read_table(shard_file_path)
                        combined_table = pa.concat_tables([existing_table, table])
                        pq.write_table(combined_table, shard_file_path, compression='snappy')
                        logger.info(f"Appended final data for {lang} to shard {global_shard_index} (total: {combined_table.num_rows} samples)")
                    else:
                        pq.write_table(table, shard_file_path, compression='snappy')
                        logger.info(f"Saved final data for {lang} to shard {global_shard_index} with {len(current_shard_data['input_ids'])} samples")
                except Exception as e:
                    logger.error(f"Error writing final shard for {lang}: {e}")
            
            # Update processed languages
            processed_languages.append(lang)
            
            # Save checkpoint
            if args.save_every_n_lines:
                import json
                checkpoint_data = {
                    'total_lines_processed': global_total_lines,
                    'last_shard_index': global_shard_index - 1,
                    'processed_languages': processed_languages
                }
                with open(checkpoint_path, 'w') as f:
                    json.dump(checkpoint_data, f)
                logger.info(f"Checkpoint saved after {lang}: {global_total_lines} total lines")
            
            logger.info(f"Completed {lang}: {lang_lines_processed} samples processed")
            log_memory(f"After {lang} completion")
            
            # Clean up this language's dataset from memory
            del raw_dataset
            del tokenized_dataset
            del lm_dataset
            gc.collect()
            log_memory(f"After {lang} cleanup")
            
        except Exception as e:
            logger.error(f"Error processing language {lang}: {e}", exc_info=True)
            continue
    
    # Final summary
    logger.info(f"\n{'='*50}")
    logger.info(f"All languages processed!")
    logger.info(f"Total lines: {global_total_lines}")
    logger.info(f"Total shards: {global_shard_index}")
    logger.info(f"Languages processed: {processed_languages}")
    logger.info(f"{'='*50}")
    log_memory("Final")
    
    # Update dataset info with final count
    if os.path.exists(dataset_info_path):
        with open(dataset_info_path, 'r') as f:
            dataset_info = json.load(f)
        dataset_info["splits"]["train"]["num_examples"] = global_total_lines
        with open(dataset_info_path, 'w') as f:
            json.dump(dataset_info, f, indent=2)
        logger.info(f"Updated dataset info with {global_total_lines} total examples")


if __name__ == "__main__":
    main()
