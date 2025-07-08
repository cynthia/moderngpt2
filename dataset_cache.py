import os
import json
import hashlib
import pickle
import shutil
from pathlib import Path
from typing import List, Dict, Optional, Union
from datasets import Dataset, concatenate_datasets, load_dataset, interleave_datasets
from transformers.utils import logging
import yaml

logger = logging.get_logger(__name__)

class DatasetCache:
    """
    A caching system for HuggingFace datasets that:
    - Caches concatenated datasets to avoid repeated processing
    - Uses fingerprinting to detect when cache needs invalidation
    - Supports both Arrow and Pickle formats for flexibility
    """
    
    def __init__(self, cache_dir: str = ".dataset_cache"):
        """
        Initialize the DatasetCache.
        
        Args:
            cache_dir: Directory to store cached datasets
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.metadata_file = self.cache_dir / "cache_metadata.json"
        self._load_metadata()
    
    def _load_metadata(self):
        """Load cache metadata from disk."""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {}
    
    def _save_metadata(self):
        """Save cache metadata to disk."""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)
    
    def _compute_fingerprint(self, files: List[str], config: Dict) -> str:
        """
        Compute a fingerprint for a dataset configuration.
        
        Args:
            files: List of file paths
            config: Additional configuration (e.g., eval_size, block_size)
        
        Returns:
            SHA256 hash string
        """
        hasher = hashlib.sha256()
        
        # Hash file paths and their modification times
        for file_path in sorted(files):
            if os.path.exists(file_path):
                stat = os.stat(file_path)
                hasher.update(f"{file_path}:{stat.st_mtime}:{stat.st_size}".encode())
            else:
                hasher.update(f"{file_path}:missing".encode())
        
        # Hash configuration
        config_str = json.dumps(config, sort_keys=True)
        hasher.update(config_str.encode())
        
        return hasher.hexdigest()
    
    def get_cached_dataset(
        self, 
        cache_key: str, 
        files: List[str], 
        config: Dict
    ) -> Optional[Dataset]:
        """
        Retrieve a cached dataset if it exists and is valid.
        
        Args:
            cache_key: Unique key for this dataset
            files: List of source files
            config: Configuration used to create the dataset
        
        Returns:
            Cached dataset if valid, None otherwise
        """
        fingerprint = self._compute_fingerprint(files, config)
        
        if cache_key in self.metadata:
            cached_info = self.metadata[cache_key]
            if cached_info['fingerprint'] == fingerprint:
                cache_path = self.cache_dir / cached_info['filename']
                if cache_path.exists():
                    try:
                        logger.info(f"Loading cached dataset from {cache_path}")
                        
                        if cached_info['format'] == 'arrow':
                            dataset = Dataset.load_from_disk(str(cache_path))
                        else:  # pickle format
                            with open(cache_path, 'rb') as f:
                                dataset = pickle.load(f)
                        
                        logger.info(f"Successfully loaded cached dataset with {len(dataset)} samples")
                        return dataset
                    except Exception as e:
                        logger.warning(f"Failed to load cached dataset: {e}")
                        # Remove corrupted cache entry
                        self._remove_cache_entry(cache_key)
        
        return None
    
    def save_dataset(
        self, 
        cache_key: str, 
        dataset: Dataset, 
        files: List[str], 
        config: Dict,
        use_arrow: bool = True
    ):
        """
        Save a dataset to cache.
        
        Args:
            cache_key: Unique key for this dataset
            dataset: Dataset to cache
            files: List of source files
            config: Configuration used to create the dataset
            use_arrow: Whether to use Arrow format (faster) or Pickle (more compatible)
        """
        fingerprint = self._compute_fingerprint(files, config)
        
        # Remove old cache if it exists
        if cache_key in self.metadata:
            self._remove_cache_entry(cache_key)
        
        # Save new dataset
        if use_arrow:
            filename = f"{cache_key}_{fingerprint[:8]}.arrow"
            cache_path = self.cache_dir / filename
            dataset.save_to_disk(str(cache_path))
            format_type = 'arrow'
        else:
            filename = f"{cache_key}_{fingerprint[:8]}.pkl"
            cache_path = self.cache_dir / filename
            with open(cache_path, 'wb') as f:
                pickle.dump(dataset, f)
            format_type = 'pickle'
        
        # Update metadata
        self.metadata[cache_key] = {
            'fingerprint': fingerprint,
            'filename': filename,
            'format': format_type,
            'files': files,
            'config': config,
            'num_samples': len(dataset)
        }
        self._save_metadata()
        
        logger.info(f"Cached dataset saved to {cache_path} ({len(dataset)} samples)")
    
    def _remove_cache_entry(self, cache_key: str):
        """Remove a cache entry and its associated files."""
        if cache_key in self.metadata:
            cache_info = self.metadata[cache_key]
            cache_path = self.cache_dir / cache_info['filename']
            
            if cache_path.exists():
                if cache_path.is_dir():
                    shutil.rmtree(cache_path)
                else:
                    cache_path.unlink()
            
            del self.metadata[cache_key]
            self._save_metadata()
    
    def clear_cache(self):
        """Clear all cached datasets."""
        logger.info("Clearing dataset cache...")
        
        for cache_key in list(self.metadata.keys()):
            self._remove_cache_entry(cache_key)
        
        logger.info("Dataset cache cleared")
    
    def get_cache_info(self) -> Dict:
        """Get information about cached datasets."""
        info = {
            'cache_dir': str(self.cache_dir),
            'num_cached_datasets': len(self.metadata),
            'total_samples': sum(entry['num_samples'] for entry in self.metadata.values()),
            'datasets': {}
        }
        
        for key, entry in self.metadata.items():
            info['datasets'][key] = {
                'num_samples': entry['num_samples'],
                'files': len(entry['files']),
                'format': entry['format']
            }
        
        return info


def load_and_concatenate_datasets_with_cache(
    files: List[str],
    cache_key: str,
    eval_size: Optional[int] = None,
    cache_dir: str = ".dataset_cache",
    streaming: bool = False,
    add_labels: bool = True,
    is_main_process: bool = True,
    accelerator=None
) -> Dataset:
    """
    Load and concatenate multiple dataset files with caching support.
    
    Args:
        files: List of Parquet file paths to load
        cache_key: Unique identifier for this dataset configuration
        eval_size: If provided, limit each file to this many samples
        cache_dir: Directory for caching
        streaming: Whether to use streaming mode (disables caching)
        add_labels: Whether to add 'labels' column (copy of 'input_ids')
    
    Returns:
        Concatenated dataset
    """
    if streaming:
        logger.info("Streaming mode enabled, skipping cache")
        datasets = []
        for file_path in files:
            if os.path.exists(file_path):
                ds = load_dataset("parquet", data_files=file_path, split="train", streaming=True)
                datasets.append(ds)
        return interleave_datasets(datasets) if len(datasets) > 1 else datasets[0]
    
    # Initialize cache
    cache = DatasetCache(cache_dir)
    
    # Configuration for fingerprinting
    config = {
        'eval_size': eval_size,
        'streaming': streaming,
        'add_labels': add_labels
    }
    
    # Try to load from cache
    cached_dataset = cache.get_cached_dataset(cache_key, files, config)
    if cached_dataset is not None:
        return cached_dataset
    
    # Only load and concatenate on the main process if cache was not found
    # This prevents multiple processes from doing the same work
    if is_main_process:
        # Load and concatenate datasets
        logger.info(f"Loading and concatenating {len(files)} dataset files...")
        datasets = []
        
        for file_path in files:
            if not os.path.exists(file_path):
                logger.warning(f"File not found, skipping: {file_path}")
                continue
            
            logger.info(f"Loading file: {file_path}")
            ds = load_dataset("parquet", data_files=file_path, split="train", streaming=False)
            
            # Apply size limit if specified
            if eval_size is not None and len(ds) > eval_size:
                ds = ds.select(range(eval_size))
            
            datasets.append(ds)
        
        if not datasets:
            raise ValueError("No valid dataset files found")
        
        # Concatenate all datasets
        logger.info("Concatenating datasets...")
        concatenated_dataset = concatenate_datasets(datasets)
        
        # Add labels column if requested and not present
        if add_labels and 'labels' not in concatenated_dataset.column_names:
            logger.info("Adding 'labels' column to concatenated dataset...")
            concatenated_dataset = concatenated_dataset.map(
                lambda x: {'labels': x['input_ids']}, 
                batched=True,
                num_proc=os.cpu_count(),
                desc="Adding labels column",
                keep_in_memory=False  # Write to disk to save memory
            )
    else:
        # Non-main processes wait for the main process
        concatenated_dataset = None
    
    # Wait for main process to finish before continuing
    if accelerator is not None:
        accelerator.wait_for_everyone()
        
    # If this is not the main process, we need to reload from cache
    if not is_main_process:
        # Give main process a moment to finish writing
        import time
        time.sleep(0.5)
        
        # Try to load from cache again
        cached_dataset = cache.get_cached_dataset(cache_key, files, config)
        if cached_dataset is not None:
            return cached_dataset
        else:
            raise RuntimeError(f"Failed to load dataset from cache on non-main process. Cache key: {cache_key}")
    
    # Save to cache (with labels column included) - only on main process
    if is_main_process:
        cache.save_dataset(cache_key, concatenated_dataset, files, config)
    
    # Wait for main process to finish saving before other processes continue
    if accelerator is not None:
        accelerator.wait_for_everyone()
    
    return concatenated_dataset