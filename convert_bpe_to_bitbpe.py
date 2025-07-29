#!/usr/bin/env python3
"""
Convert a standard BPE tokenizer to BitBPE format with UTF-8 bit redistribution.

This script takes an existing BPE tokenizer (with byte-level fallback) and modifies it
to support the BitBPE encoding scheme from the paper "Breaking the Curse of Multilinguality
with Cross-lingual Expert Language Models" (https://arxiv.org/html/2506.07541v1).

The BitBPE approach redistributes bits in 3-byte UTF-8 sequences (common in CJK languages)
to achieve better compression by using 6-bit prefixes and two 9-bit tokens.
"""

import argparse
import os
import json
import shutil
import sentencepiece as spm
from transformers import AutoTokenizer
from transformers.utils import logging

logger = logging.get_logger(__name__)


def convert_bpe_to_bitbpe(input_path: str, output_path: str, num_prefix_tokens: int = 16):
    """
    Convert a BPE tokenizer to BitBPE format.
    
    Args:
        input_path: Path to the original BPE tokenizer directory
        output_path: Path to save the BitBPE tokenizer
        num_prefix_tokens: Number of tokens to reserve for BitBPE prefixes (default: 16 for 0xE0-0xEF)
    """
    logger.info(f"Converting BPE tokenizer from {input_path} to BitBPE format")
    
    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    
    # Copy all files from input to output
    for filename in os.listdir(input_path):
        src_file = os.path.join(input_path, filename)
        dst_file = os.path.join(output_path, filename)
        if os.path.isfile(src_file):
            shutil.copy2(src_file, dst_file)
            logger.info(f"Copied {filename}")
    
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(input_path, trust_remote_code=True)
    original_vocab_size = len(tokenizer)
    logger.info(f"Original tokenizer vocabulary size: {original_vocab_size}")
    
    # Modify tokenizer_config.json to indicate BitBPE mode
    config_path = os.path.join(output_path, "tokenizer_config.json")
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Add BitBPE configuration
        # Note: We need to account for the actual token IDs used
        # The tokenizer will create prefix tokens and 512 9-bit tokens
        config['tokenizer_type'] = 'bitbpe'
        config['bitbpe_config'] = {
            'enabled': True,
            'num_prefix_tokens': num_prefix_tokens,
            'prefix_token_start_id': original_vocab_size + 2,  # Start after existing tokens
            'version': '1.0'
        }
        
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        logger.info("Updated tokenizer_config.json with BitBPE configuration")
    
    # Load and modify the SentencePiece model
    sp_model_path = os.path.join(input_path, "tokenizer.model")
    if os.path.exists(sp_model_path):
        # For now, we'll copy the model as-is and rely on the tokenizer implementation
        # to handle BitBPE prefix tokens virtually
        output_model_path = os.path.join(output_path, "tokenizer.model")
        shutil.copy2(sp_model_path, output_model_path)
        
        logger.info(f"SentencePiece model copied. BitBPE will use {num_prefix_tokens} prefix tokens for byte fallback optimization")
        logger.info(f"BitBPE optimizes 3-byte UTF-8 sequences (0xE0-0xEF) in byte fallback mode")
    
    # Update tokenizer.json to include BitBPE tokens in the vocabulary
    tokenizer_json_path = os.path.join(output_path, "tokenizer.json")
    if os.path.exists(tokenizer_json_path):
        with open(tokenizer_json_path, 'r') as f:
            tokenizer_data = json.load(f)
        
        # Add BitBPE tokens to the vocabulary
        if 'model' in tokenizer_data and 'vocab' in tokenizer_data['model']:
            vocab = tokenizer_data['model']['vocab']
            
            # Get the current max token ID
            max_token_id = max(vocab.values()) if vocab else -1
            next_id = max_token_id + 1
            
            # Add prefix tokens (spaced out as in the actual tokenizer)
            # Based on the actual implementation, these are at specific IDs
            prefix_start = original_vocab_size + 2
            prefix_ids = [prefix_start + i * 4 + 3 for i in range(4)]  # 8005, 8009, 8013, 8017
            
            # Add 9-bit tokens starting after the last prefix token
            bit9_start = prefix_ids[-1] + 1  # 8018
            
            # Note: We'll document the expected token IDs but won't modify the vocab
            # as the tokenizer implementation will handle these virtually
            logger.info(f"BitBPE token allocation:")
            logger.info(f"  - Prefix tokens: {prefix_ids} (for prefixes 0x38-0x3B)")
            logger.info(f"  - 9-bit tokens: {bit9_start} to {bit9_start + 511}")
            
            # Update the config with the actual token allocations
            config = None
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            config['bitbpe_config']['prefix_token_ids'] = prefix_ids
            config['bitbpe_config']['bit9_start_id'] = bit9_start
            
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
    
    # Create or update special_tokens_map.json
    special_tokens_path = os.path.join(output_path, "special_tokens_map.json")
    if os.path.exists(special_tokens_path):
        with open(special_tokens_path, 'r') as f:
            special_tokens = json.load(f)
    else:
        special_tokens = {}
    
    with open(special_tokens_path, 'w') as f:
        json.dump(special_tokens, f, indent=2)
    
    # Create a metadata file for BitBPE conversion info
    metadata = {
        'conversion_info': {
            'original_tokenizer': input_path,
            'original_vocab_size': original_vocab_size,
            'bitbpe_vocab_size': original_vocab_size + num_prefix_tokens + 512,  # Add 512 for 9-bit tokens
            'num_prefix_tokens': num_prefix_tokens,
            'description': 'BitBPE tokenizer optimizing byte fallback for 3-byte UTF-8 sequences'
        },
        'bit_redistribution_rules': {
            '3_byte_utf8': {
                'description': 'For 3-byte UTF-8 sequences (0xE0-0xEF)',
                'encoding': [
                    '6-bit prefix token',
                    '9-bit token (2 bits from byte1 + 7 bits from byte2)',
                    '9-bit token (1 bit from byte2 + 8 bits from byte3)'
                ]
            }
        }
    }
    
    metadata_path = os.path.join(output_path, "bitbpe_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"BitBPE conversion complete! Tokenizer saved to {output_path}")
    
    # Test the converted tokenizer
    logger.info("Testing the converted tokenizer...")
    try:
        test_tokenizer = AutoTokenizer.from_pretrained(output_path, trust_remote_code=True)
        test_text = "Hello 世界! こんにちは 안녕하세요"
        tokens = test_tokenizer.tokenize(test_text)
        logger.info(f"Test tokenization successful: {len(tokens)} tokens")
        
        # Test with BitBPE mode if supported
        if hasattr(test_tokenizer, 'set_tokenizer_type'):
            test_tokenizer.set_tokenizer_type('bitbpe')
            bitbpe_tokens = test_tokenizer.tokenize(test_text)
            logger.info(f"BitBPE tokenization: {len(bitbpe_tokens)} tokens")
    except Exception as e:
        logger.error(f"Error testing converted tokenizer: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert a BPE tokenizer to BitBPE format with UTF-8 bit redistribution"
    )
    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="Path to the original BPE tokenizer directory"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to save the BitBPE tokenizer"
    )
    parser.add_argument(
        "--num_prefix_tokens",
        type=int,
        default=16,
        help="Number of tokens to reserve for BitBPE prefixes (default: 16 for UTF-8 prefixes 0xE0-0xEF)"
    )
    parser.add_argument(
        "--test_conversion",
        action="store_true",
        help="Run tests on the converted tokenizer"
    )
    
    args = parser.parse_args()
    
    # Set up logging
    logging.set_verbosity_info()
    
    # Perform conversion
    convert_bpe_to_bitbpe(args.input_path, args.output_path, args.num_prefix_tokens)
    
    if args.test_conversion:
        logger.info("\nRunning additional tests on the converted tokenizer...")
        
        # Test with various CJK texts
        test_texts = [
            "The quick brown fox jumps over the lazy dog",  # English
            "これは日本語のテストです。",  # Japanese
            "这是中文测试。",  # Simplified Chinese
            "這是中文測試。",  # Traditional Chinese
            "한국어 테스트입니다.",  # Korean
            "Mixed: Hello 世界! Bonjour こんにちは!"  # Mixed languages
        ]
        
        original_tokenizer = AutoTokenizer.from_pretrained(args.input_path, trust_remote_code=True)
        bitbpe_tokenizer = AutoTokenizer.from_pretrained(args.output_path, trust_remote_code=True)
        
        for text in test_texts:
            orig_tokens = original_tokenizer.tokenize(text)
            bitbpe_tokens = bitbpe_tokenizer.tokenize(text)
            
            logger.info(f"\nText: {text}")
            logger.info(f"Original BPE: {len(orig_tokens)} tokens")
            logger.info(f"BitBPE: {len(bitbpe_tokens)} tokens")
            
            # Check if text can be correctly encoded and decoded
            orig_ids = original_tokenizer.encode(text, add_special_tokens=False)
            bitbpe_ids = bitbpe_tokenizer.encode(text, add_special_tokens=False)
            
            orig_decoded = original_tokenizer.decode(orig_ids)
            bitbpe_decoded = bitbpe_tokenizer.decode(bitbpe_ids)
            
            if orig_decoded == bitbpe_decoded == text:
                logger.info("✓ Encoding/decoding test passed")
            else:
                logger.warning("✗ Encoding/decoding mismatch detected")


if __name__ == "__main__":
    main()