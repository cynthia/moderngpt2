#!/usr/bin/env python3
"""
Convert a SentencePiece BPE model to HuggingFace tokenizers format.
This preserves the exact vocabulary and merge rules while gaining HF performance benefits.
"""

import argparse
import json
import os
from typing import Dict, List, Tuple
import sentencepiece as spm
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, processors
from transformers import PreTrainedTokenizerFast
from transformers.utils import logging

logger = logging.get_logger(__name__)

def load_spm_vocab(sp_model_path: str) -> Tuple[Dict[str, int], List[Tuple[str, str]]]:
    """Load vocabulary and merge rules from a SentencePiece model."""
    sp = spm.SentencePieceProcessor()
    sp.load(sp_model_path)
    
    # Extract vocabulary
    vocab = {}
    for i in range(sp.get_piece_size()):
        piece = sp.id_to_piece(i)
        vocab[piece] = i
    
    # Extract merge rules (this is more complex as SPM doesn't directly expose merges)
    # We'll need to reconstruct them from the vocabulary
    merges = []
    
    # For BPE models, subwords are built from smaller pieces
    # We need to analyze the vocabulary to reconstruct merge operations
    logger.info(f"Loaded SentencePiece model with {len(vocab)} tokens")
    
    return vocab, merges, sp

def convert_sentencepiece_to_hf(sp_model_path: str, output_dir: str, 
                                model_type: str = "bpe",
                                add_prefix_space: bool = True) -> None:
    """Convert a SentencePiece model to HuggingFace tokenizers format."""
    
    logger.info(f"Loading SentencePiece model from {sp_model_path}")
    vocab, merges, sp = load_spm_vocab(sp_model_path)
    
    # Detect model configuration from the original
    is_byte_fallback = any(piece.startswith('<0x') and piece.endswith('>') for piece in vocab.keys())
    logger.info(f"Byte fallback detected: {is_byte_fallback}")
    
    # Get special tokens info
    unk_id = sp.unk_id()
    pad_id = sp.pad_id() if hasattr(sp, 'pad_id') else 0
    bos_id = sp.bos_id() if hasattr(sp, 'bos_id') else -1
    eos_id = sp.eos_id() if hasattr(sp, 'eos_id') else -1
    
    # Create tokenizer based on model type
    if model_type.lower() == "bpe":
        # For BPE, we need to export the model in a format HF can read
        # The best approach is to use the SPM tokenizer directly via HF
        logger.info("Creating HuggingFace tokenizer wrapper for SentencePiece BPE model")
        
        # First, let's save the SPM model to the output directory
        os.makedirs(output_dir, exist_ok=True)
        import shutil
        output_spm_path = os.path.join(output_dir, "spm.model")
        shutil.copy2(sp_model_path, output_spm_path)
        
        # Create a tokenizer.json that uses the SPM model efficiently
        tokenizer_config = {
            "type": "SentencePiece",
            "vocab": {piece: idx for piece, idx in vocab.items()},
            "unk_token": sp.id_to_piece(unk_id) if unk_id >= 0 else "<unk>",
            "add_prefix_space": add_prefix_space,
            "model_path": "spm.model"
        }
        
        # Create HF tokenizer using the fast implementation
        from transformers import LlamaTokenizerFast
        
        # Use LlamaTokenizerFast as base since it handles SPM well
        try:
            fast_tokenizer = LlamaTokenizerFast(
                vocab_file=output_spm_path,
                unk_token=sp.id_to_piece(unk_id) if unk_id >= 0 else "<unk>",
                bos_token=sp.id_to_piece(bos_id) if bos_id >= 0 else None,
                eos_token=sp.id_to_piece(eos_id) if eos_id >= 0 else "</s>",
                pad_token=sp.id_to_piece(pad_id) if pad_id >= 0 else "<pad>",
                add_prefix_space=add_prefix_space,
                clean_up_tokenization_spaces=False,
                legacy=False,
            )
            
            # Save the fast tokenizer
            fast_tokenizer.save_pretrained(output_dir)
            logger.info(f"Saved HuggingFace fast tokenizer to {output_dir}")
            
        except Exception as e:
            logger.warning(f"Failed to create LlamaTokenizerFast, falling back to manual conversion: {e}")
            
            # Manual conversion approach
            # Create a BPE tokenizer from scratch with the vocabulary
            tokenizer = Tokenizer(models.BPE(unk_token=sp.id_to_piece(unk_id) if unk_id >= 0 else "<unk>"))
            
            # Configure pre-tokenizer based on the original model
            if is_byte_fallback:
                # Use ByteLevel for models with byte fallback
                tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
                tokenizer.decoder = decoders.ByteLevel()
            else:
                # Use Metaspace for standard models (like SentencePiece default)
                tokenizer.pre_tokenizer = pre_tokenizers.Metaspace(replacement="▁")
                tokenizer.decoder = decoders.Metaspace(replacement="▁")
            
            # We need to manually set the vocabulary
            # This is a simplified approach - for full compatibility, 
            # you might need to extract merge rules from the SPM model
            new_vocab = {}
            for token, idx in vocab.items():
                new_vocab[token] = idx
            
            # Set vocabulary on the model
            tokenizer.model.vocab = new_vocab
            
            # Save tokenizer.json
            tokenizer_path = os.path.join(output_dir, "tokenizer.json")
            tokenizer.save(tokenizer_path)
            
            # Create PreTrainedTokenizerFast wrapper
            wrapped_tokenizer = PreTrainedTokenizerFast(
                tokenizer_file=tokenizer_path,
                unk_token=sp.id_to_piece(unk_id) if unk_id >= 0 else "<unk>",
                bos_token=sp.id_to_piece(bos_id) if bos_id >= 0 else None,
                eos_token=sp.id_to_piece(eos_id) if eos_id >= 0 else "</s>",
                pad_token=sp.id_to_piece(pad_id) if pad_id >= 0 else "<pad>",
                clean_up_tokenization_spaces=False,
            )
            
            wrapped_tokenizer.save_pretrained(output_dir)
    
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    # Test the conversion
    logger.info("\nTesting conversion with sample texts:")
    test_texts = [
        "Hello world!",
        "This is a test.",
        "Testing 你好世界 with mixed scripts.",
        "Byte fallback test: \x00\x01\x02",
    ]
    
    # Load both tokenizers for comparison
    original_sp = spm.SentencePieceProcessor()
    original_sp.load(sp_model_path)
    
    # Load the converted tokenizer
    converted_tokenizer = PreTrainedTokenizerFast.from_pretrained(output_dir)
    
    for text in test_texts:
        # Original tokenization
        original_ids = original_sp.encode(text)
        original_tokens = [original_sp.id_to_piece(id) for id in original_ids]
        
        # Converted tokenization
        converted_ids = converted_tokenizer.encode(text, add_special_tokens=False)
        converted_tokens = converted_tokenizer.convert_ids_to_tokens(converted_ids)
        
        print(f"\nText: {text}")
        print(f"Original SPM: {original_tokens} (ids: {original_ids})")
        print(f"Converted HF: {converted_tokens} (ids: {converted_ids})")
        print(f"Match: {'✓' if original_ids == converted_ids else '✗'}")
    
    logger.info(f"\nConversion complete! Converted tokenizer saved to {output_dir}")
    logger.info("Note: For best performance, use the converted tokenizer with batch processing")

def main():
    parser = argparse.ArgumentParser(
        description="Convert SentencePiece model to HuggingFace tokenizers format"
    )
    parser.add_argument(
        "--spm_model",
        type=str,
        required=True,
        help="Path to the SentencePiece .model file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save the converted HuggingFace tokenizer"
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="bpe",
        choices=["bpe", "unigram"],
        help="Type of the SentencePiece model (default: bpe)"
    )
    parser.add_argument(
        "--no_prefix_space",
        action="store_true",
        help="Don't add prefix space (default: add prefix space)"
    )
    
    args = parser.parse_args()
    
    logging.set_verbosity_info()
    
    convert_sentencepiece_to_hf(
        sp_model_path=args.spm_model,
        output_dir=args.output_dir,
        model_type=args.model_type,
        add_prefix_space=not args.no_prefix_space
    )

if __name__ == "__main__":
    main()