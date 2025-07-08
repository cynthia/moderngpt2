import torch
from itertools import islice
from datasets import load_dataset, interleave_datasets, concatenate_datasets
from transformers import AutoTokenizer
from transformers.utils import logging
import glob # Add glob for finding parquet files
import os # Add os for path join
import yaml # Add yaml for parsing metadata files
from dataset_cache import DatasetCache, load_and_concatenate_datasets_with_cache

logger = logging.get_logger(__name__)

def get_dataset(tokenizer_path: str, block_size: int, streaming: bool = True, pre_tokenized_path: str = None, streaming_eval_samples: int = 1000, metadata_file: str = None, cache_dir: str = ".dataset_cache", use_cache: bool = True):
    logger.info(f"Loading tokenizer from: {tokenizer_path}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    except Exception as e:
        logger.error(f"Could not load tokenizer from {tokenizer_path}. Error: {e}")
        raise
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if metadata_file:
        logger.info(f"Loading dataset configuration from metadata file: {metadata_file}")
        try:
            with open(metadata_file, 'r') as f:
                metadata = yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Failed to load metadata file {metadata_file}: {e}")
            raise
        
        # Extract configuration
        train_files = metadata.get('train', [])
        eval_files = metadata.get('eval', [])
        eval_size = metadata.get('eval_size', 10000)
        
        if not train_files:
            raise ValueError("No training files specified in metadata file")
        if not eval_files:
            raise ValueError("No evaluation files specified in metadata file")
        
        logger.info(f"Found {len(train_files)} training files and {len(eval_files)} evaluation files")
        logger.info(f"Evaluation size per file: {eval_size}")
        
        # Load training datasets with caching support
        if use_cache and not streaming:
            train_dataset = load_and_concatenate_datasets_with_cache(
                files=train_files,
                cache_key=f"train_{os.path.basename(metadata_file)}",
                eval_size=None,
                cache_dir=cache_dir,
                streaming=False,
                add_labels=True  # Handle labels in the caching function
            )
            logger.info(f"Total training samples: {len(train_dataset)}")
        else:
            # Original non-cached loading
            train_datasets = []
            for file_path in train_files:
                if not os.path.exists(file_path):
                    logger.warning(f"Training file not found, skipping: {file_path}")
                    continue
                logger.info(f"Loading training file: {file_path}")
                ds = load_dataset("parquet", data_files=file_path, split="train", streaming=False)
                train_datasets.append(ds)
            
            if not train_datasets:
                raise ValueError("No valid training files found")
            
            # Concatenate all training datasets
            train_dataset = concatenate_datasets(train_datasets)
            logger.info(f"Total training samples: {len(train_dataset)}")
        
        # Load evaluation datasets with caching support
        if use_cache and not streaming:
            eval_dataset = load_and_concatenate_datasets_with_cache(
                files=eval_files,
                cache_key=f"eval_{os.path.basename(metadata_file)}",
                eval_size=eval_size,
                cache_dir=cache_dir,
                streaming=False,
                add_labels=True  # Handle labels in the caching function
            )
            logger.info(f"Total evaluation samples: {len(eval_dataset)}")
        else:
            # Original non-cached loading
            eval_datasets = []
            for file_path in eval_files:
                if not os.path.exists(file_path):
                    logger.warning(f"Evaluation file not found, skipping: {file_path}")
                    continue
                logger.info(f"Loading evaluation file: {file_path}")
                ds = load_dataset("parquet", data_files=file_path, split="train", streaming=False)
                # Take only eval_size samples from each file
                if len(ds) > eval_size:
                    ds = ds.select(range(eval_size))
                eval_datasets.append(ds)
            
            if not eval_datasets:
                raise ValueError("No valid evaluation files found")
            
            # Concatenate all evaluation datasets
            eval_dataset = concatenate_datasets(eval_datasets)
            logger.info(f"Total evaluation samples: {len(eval_dataset)}")
        
        # Verify columns
        if 'input_ids' not in train_dataset.column_names or 'attention_mask' not in train_dataset.column_names:
            raise ValueError(f"Training dataset must have 'input_ids' and 'attention_mask' columns. Found: {train_dataset.column_names}")
        
        # Add labels column if not present - optimized version
        # Note: When using cache, labels should already be added by the caching function
        if 'labels' not in train_dataset.column_names:
            logger.info("Adding 'labels' column to training dataset...")
            train_dataset = train_dataset.map(
                lambda x: {'labels': x['input_ids']}, 
                batched=True,
                num_proc=os.cpu_count(),  # Use all available CPUs
                desc="Adding labels to train dataset",
                keep_in_memory=True  # Keep in memory for faster processing
            )
        if 'labels' not in eval_dataset.column_names:
            logger.info("Adding 'labels' column to evaluation dataset...")
            eval_dataset = eval_dataset.map(
                lambda x: {'labels': x['input_ids']}, 
                batched=True,
                num_proc=os.cpu_count(),  # Use all available CPUs
                desc="Adding labels to eval dataset",
                keep_in_memory=True  # Keep in memory for faster processing
            )
        
    elif pre_tokenized_path:
        logger.info(f"Attempting to load pre-tokenized dataset from: {pre_tokenized_path}")
        parquet_files = glob.glob(os.path.join(pre_tokenized_path, "*.parquet"))

        if not parquet_files:
            logger.error(f"No .parquet files found in {pre_tokenized_path}. Please ensure the path is correct and files exist.")
            raise FileNotFoundError(f"No .parquet files found in {pre_tokenized_path}")

        logger.info(f"Found {len(parquet_files)} Parquet files: {parquet_files}")

        # Load the dataset from Parquet files.
        # `streaming=True` can be used if the dataset library supports efficient streaming from multiple Parquet files.
        # If not streaming, the entire dataset will be loaded into memory, which might be large.
        train_dataset = load_dataset("parquet", data_files=parquet_files, split="train", streaming=streaming)

        logger.info(f"Pre-tokenized dataset loaded. Columns: {train_dataset.column_names if not streaming else 'unknown (streaming)'}")

        # Ensure required columns are present (example, actual check might be more robust)
        # For streaming, column names might not be immediately available without iterating.
        # We assume the pretokenized files are correctly formatted.

        if streaming:
            logger.info(f"Taking {streaming_eval_samples} samples from the streaming pre-tokenized train dataset for evaluation.")
            eval_dataset = train_dataset.take(streaming_eval_samples)
        else:
            # If not streaming, we might need a different way to split, or just take a slice.
            # For simplicity, if it's a non-streaming HF dataset, .select() can be used.
            # However, load_dataset with split='train' gives a Dataset object.
            # To create a small eval set from a non-streaming dataset:
            logger.info(f"Creating evaluation dataset from non-streaming pre-tokenized data (first {streaming_eval_samples} samples).")
            # This creates an iterable dataset; convert to list if needed by trainer, but trainer usually handles iterables.
            eval_dataset = train_dataset.select(range(min(streaming_eval_samples, len(train_dataset))))


    else: # Original logic for on-the-fly tokenization
        logger.info("Loading and processing C4 dataset in streaming mode (pre_tokenized_path not provided)...")
        langs = ["en", "ja", "ko", "zh"]
        datasets_list = [load_dataset("allenai/c4", lang, streaming=True, split="train", trust_remote_code=True) for lang in langs]
        interleaved_ds = interleave_datasets(datasets_list)

        def tokenize_function(examples):
            text = examples["text"]
            if not isinstance(text, str) or not text.strip():
                return {"input_ids": [], "attention_mask": []}
            return tokenizer(text, truncation=False)

        tokenized_ds = interleaved_ds.map(
            tokenize_function,
            batched=True,
            remove_columns=["text", "timestamp", "url"]
        )

        def group_texts(examples):
            concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
            total_length = len(concatenated_examples[list(examples.keys())[0]])
            if total_length == 0:
                 return {k: [] for k in examples.keys()}
            if total_length >= block_size:
                total_length = (total_length // block_size) * block_size
            else:
                return {k: [] for k in examples.keys()}
            result = {
                k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
                for k, t in concatenated_examples.items()
            }
            result["labels"] = result["input_ids"].copy()
            return result

        logger.info(f"Grouping texts into blocks of size {block_size}...")
        lm_dataset = tokenized_ds.map(group_texts, batched=True).filter(lambda x: len(x['input_ids']) > 0)

        train_dataset = lm_dataset
        logger.info(f"Taking {streaming_eval_samples} samples from the streaming train dataset for evaluation.")
        eval_dataset = train_dataset.take(streaming_eval_samples)

    return train_dataset, eval_dataset, tokenizer

if __name__ == '__main__':
    import tempfile
    import pyarrow.parquet as pq
    import pyarrow as pa
    import shutil

    # Common setup for tokenizer
    tokenizer_directory = "dummy_tokenizer_for_dataset_testing"
    try:
        AutoTokenizer.from_pretrained(tokenizer_directory, trust_remote_code=True)
        logger.info(f"Found existing tokenizer at {tokenizer_directory}")
    except Exception:
        logger.info(f"Tokenizer not found at {tokenizer_directory}. Creating a dummy gpt2 tokenizer for testing.")
        try:
            temp_tokenizer = AutoTokenizer.from_pretrained("gpt2")
            temp_tokenizer.save_pretrained(tokenizer_directory) # Saves tokenizer.json, etc.
            logger.info(f"Dummy tokenizer saved to {tokenizer_directory}")
        except Exception as e:
            logger.error(f"Could not create dummy tokenizer: {e}.")
            raise

    # Test 1: Original streaming C4 dataset
    print("\n--- Testing with streaming C4 dataset ---")
    try:
        train_ds_c4, eval_ds_c4, tok_c4 = get_dataset(
            tokenizer_path=tokenizer_directory,
            block_size=512, # Smaller block for faster test
            streaming=True,
            streaming_eval_samples=2
        )
        print(f"C4: Train dataset type: {type(train_ds_c4)}, Eval dataset type: {type(eval_ds_c4)}")
        print("C4: First sample from train_ds:", next(iter(train_ds_c4)))
        print("C4: First sample from eval_ds:", next(iter(eval_ds_c4)))
    except Exception as e:
        logger.error(f"Error during C4 streaming test: {e}")
        print(f"C4 streaming test failed: {e}")

    # Test 2: Pre-tokenized Parquet dataset
    print("\n--- Testing with pre-tokenized Parquet dataset ---")
    # Create a temporary directory for dummy parquet files
    temp_parquet_dir = tempfile.mkdtemp(prefix="dummy_parquet_")
    try:
        # Create some dummy pre-tokenized data
        dummy_data = {
            "input_ids": [[i for i in range(10)] for _ in range(5)], # 5 samples, sequence length 10
            "attention_mask": [[1]*10 for _ in range(5)],
            "labels": [[i for i in range(10)] for _ in range(5)],
        }
        table = pa.Table.from_pydict(dummy_data)
        dummy_parquet_file = os.path.join(temp_parquet_dir, "dummy_shard_0.parquet")
        pq.write_table(table, dummy_parquet_file)
        print(f"Created dummy Parquet file: {dummy_parquet_file}")

        # Test get_dataset with pre_tokenized_path (streaming)
        print("\n--- Testing Parquet (streaming=True) ---")
        train_ds_pq_stream, eval_ds_pq_stream, tok_pq_stream = get_dataset(
            tokenizer_path=tokenizer_directory,
            block_size=1024, # block_size is not used for pretokenized, but arg is required
            pre_tokenized_path=temp_parquet_dir,
            streaming=True,
            streaming_eval_samples=2
        )
        print(f"Parquet (Stream): Train dataset type: {type(train_ds_pq_stream)}, Eval dataset type: {type(eval_ds_pq_stream)}")
        print("Parquet (Stream): First sample from train_ds:", next(iter(train_ds_pq_stream)))
        # Note: eval_ds_pq_stream is an iterable, taking from train_ds_pq_stream.
        # If train_ds_pq_stream is exhausted, eval_ds_pq_stream might be empty.
        # Re-initialize or be careful with iterators. For this test, it should be fine as take() creates a new iterable.
        eval_samples_pq_stream = list(islice(eval_ds_pq_stream, 2))
        print(f"Parquet (Stream): Eval samples count: {len(eval_samples_pq_stream)}")
        if eval_samples_pq_stream:
            print("Parquet (Stream): First sample from eval_ds:", eval_samples_pq_stream[0])


        # Test get_dataset with pre_tokenized_path (streaming=False)
        print("\n--- Testing Parquet (streaming=False) ---")
        train_ds_pq_nostream, eval_ds_pq_nostream, tok_pq_nostream = get_dataset(
            tokenizer_path=tokenizer_directory,
            block_size=1024,
            pre_tokenized_path=temp_parquet_dir,
            streaming=False, # Test non-streaming load
            streaming_eval_samples=2
        )
        print(f"Parquet (NoStream): Train dataset type: {type(train_ds_pq_nostream)}, Eval dataset type: {type(eval_ds_pq_nostream)}")
        print(f"Parquet (NoStream): Train dataset length: {len(train_ds_pq_nostream)}")
        print(f"Parquet (NoStream): Eval dataset length: {len(eval_ds_pq_nostream)}")
        print("Parquet (NoStream): First sample from train_ds:", train_ds_pq_nostream[0])
        print("Parquet (NoStream): First sample from eval_ds:", eval_ds_pq_nostream[0])


    except Exception as e:
        logger.error(f"Error during Parquet loading test: {e}")
        print(f"Parquet loading test failed: {e}")
    finally:
        # Clean up dummy parquet directory
        if os.path.exists(temp_parquet_dir):
            shutil.rmtree(temp_parquet_dir)
            print(f"Cleaned up temporary Parquet directory: {temp_parquet_dir}")

    print("\nDataset module test complete (including Parquet).")
