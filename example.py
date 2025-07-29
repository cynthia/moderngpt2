#!/usr/bin/env python3

import torch
from transformers import AutoTokenizer
from moderngpt2.modeling_moderngpt2 import ModernGPT2LMHeadModel
from moderngpt2 import ModernGPT2Tokenizer
from typing import Optional, List, Union
import warnings
import os
import json

class ModernGPT2Loader:
    """Wrapper class for loading and using ModernGPT2 models."""
    
    def __init__(self, model_path: str = "output/final/8k-unk/2B", device: str = "cpu"):
        """
        Initialize the model loader.
        
        Args:
            model_path: Path to the pretrained model
            device: Device to load model on ('cpu', 'cuda', 'cuda:0', etc.)
        """
        # Handle device string properly
        if device.isdigit():
            self.device = torch.device(f"cuda:{device}")
        else:
            self.device = torch.device(device)
        self.model_path = model_path
        
        print(f"Loading model from {model_path}...")
        self.model = ModernGPT2LMHeadModel.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        
        print(f"Loading tokenizer from {model_path}...")
        
        # Check if this is a BitBPE model
        is_bitbpe = False
        bitbpe_tokenizer_path = None
        tokenizer_config_path = os.path.join(model_path, "tokenizer_config.json")
        
        if os.path.exists(tokenizer_config_path):
            with open(tokenizer_config_path, 'r') as f:
                config = json.load(f)
                if config.get("tokenizer_class") == "ModernGPT2Tokenizer" and config.get("tokenizer_type") == "bitbpe":
                    is_bitbpe = True
        
        # For BitBPE models, check if we need to use a different tokenizer path
        if is_bitbpe:
            # Check if the model path contains the correct BitBPE tokenizer files
            if not os.path.exists(os.path.join(model_path, "bitbpe_metadata.json")):
                # Look for the correct tokenizer in standard locations
                possible_paths = [
                    "model/bpe-8k-bitbpe-fixed",  # Known good BitBPE tokenizer
                    os.path.join(os.path.dirname(model_path), "tokenizer"),  # Sibling directory
                    os.path.join(os.path.dirname(os.path.dirname(model_path)), "tokenizer"),  # Parent's sibling
                ]
                
                for path in possible_paths:
                    if os.path.exists(path) and os.path.exists(os.path.join(path, "bitbpe_metadata.json")):
                        bitbpe_tokenizer_path = path
                        print(f"Note: Using BitBPE tokenizer from {path} for vocab size compatibility")
                        break
        
        # Load appropriate tokenizer
        if is_bitbpe:
            print("Detected BitBPE model, using ModernGPT2Tokenizer...")
            tokenizer_path = bitbpe_tokenizer_path or model_path
            self.tokenizer = ModernGPT2Tokenizer.from_pretrained(tokenizer_path)
        else:
            print("Using standard AutoTokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Set pad token if not already set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Get max position embeddings from model config
        self.max_position_embeddings = getattr(self.model.config, 'max_position_embeddings', 2048)
        n_positions = getattr(self.model.config, 'n_positions', self.max_position_embeddings)
        self.actual_max_pos = min(self.max_position_embeddings, n_positions)
        
        print(f"Model loaded successfully on {self.device}")
        print(f"Max position embeddings: {self.actual_max_pos}")
        print(f"Model vocab size: {self.model.config.vocab_size}")
        # For BitBPE tokenizers, use vocab_size property instead of len()
        tokenizer_vocab_size = getattr(self.tokenizer, 'vocab_size', len(self.tokenizer))
        print(f"Tokenizer vocab size: {tokenizer_vocab_size}")
        
        # Check for vocab mismatch
        if tokenizer_vocab_size != self.model.config.vocab_size:
            warnings.warn(f"Tokenizer vocab size ({tokenizer_vocab_size}) != Model vocab size ({self.model.config.vocab_size})")
            self.effective_vocab_size = min(tokenizer_vocab_size, self.model.config.vocab_size)
        else:
            self.effective_vocab_size = self.model.config.vocab_size
    
    def generate(
        self,
        prompt: str,
        max_length: int = 100,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.95,
        num_return_sequences: int = 1,
        do_sample: bool = True,
        repetition_penalty: float = 1.0,
    ) -> Union[str, List[str]]:
        """
        Generate text completion for a given prompt.
        
        Args:
            prompt: Input text prompt
            max_length: Maximum length of generated text
            temperature: Sampling temperature (higher = more random)
            top_k: Limit sampling to top k tokens
            top_p: Nucleus sampling threshold
            num_return_sequences: Number of sequences to generate
            do_sample: Whether to use sampling (vs greedy decoding)
            repetition_penalty: Penalty for repeating tokens
            
        Returns:
            Generated text (single string if num_return_sequences=1, list otherwise)
        """
        # Ensure max_length doesn't exceed model's limit
        safe_max_length = min(max_length, self.actual_max_pos)
        if safe_max_length < max_length:
            warnings.warn(f"Reducing max_length from {max_length} to {safe_max_length} due to model's position embedding limit")
        
        # Encode the prompt - handle different tokenizer return types
        tokenizer_output = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=safe_max_length)
        
        # Handle different return types from tokenizers
        if isinstance(tokenizer_output, dict):
            # Convert lists to tensors if needed
            inputs = {}
            for k, v in tokenizer_output.items():
                if isinstance(v, list):
                    # Convert list to tensor
                    inputs[k] = torch.tensor([v], dtype=torch.long).to(self.device)
                elif torch.is_tensor(v):
                    inputs[k] = v.to(self.device)
                else:
                    # Skip non-tensor, non-list items
                    continue
        elif hasattr(tokenizer_output, 'input_ids'):
            # Standard transformers BatchEncoding
            inputs = {k: v.to(self.device) for k, v in tokenizer_output.items() if torch.is_tensor(v)}
        else:
            # Fallback - assume it's just input_ids
            if isinstance(tokenizer_output, list):
                tokenizer_output = torch.tensor([tokenizer_output], dtype=torch.long)
            inputs = {'input_ids': tokenizer_output.to(self.device)}
        
        # Check for out-of-bounds tokens and fix if necessary
        if 'input_ids' in inputs:
            max_token_id = inputs['input_ids'].max().item()
            if max_token_id >= self.model.config.vocab_size:
                warnings.warn(f"Input contains out-of-bounds token ID {max_token_id} (vocab size: {self.model.config.vocab_size})")
                # Replace out-of-bounds tokens with unk_token
                unk_token_id = self.tokenizer.unk_token_id if self.tokenizer.unk_token_id is not None else 0
                inputs['input_ids'] = torch.where(
                    inputs['input_ids'] >= self.model.config.vocab_size,
                    torch.tensor(unk_token_id, device=inputs['input_ids'].device),
                    inputs['input_ids']
                )
        
        # Generate with the model
        with torch.no_grad():
            # For models with vocab mismatch, ensure we don't generate out-of-bounds tokens
            generation_kwargs = {
                **inputs,
                'max_length': safe_max_length,
                'temperature': temperature,
                'top_k': min(top_k, self.effective_vocab_size) if top_k else None,
                'top_p': top_p,
                'num_return_sequences': num_return_sequences,
                'do_sample': do_sample,
                'repetition_penalty': repetition_penalty,
                'pad_token_id': self.tokenizer.pad_token_id,
                'eos_token_id': self.tokenizer.eos_token_id,
            }
            
            # If there's a vocab size mismatch, we need to be more careful
            if self.effective_vocab_size < self.model.config.vocab_size:
                # Force top_k to prevent generating out-of-bounds tokens
                generation_kwargs['top_k'] = min(50, self.effective_vocab_size)
                generation_kwargs['do_sample'] = True  # Ensure sampling mode
                
                # WORKAROUND: For BitBPE models with vocab issues during generation,
                # there's a known issue where the model may generate token IDs that
                # exceed the embedding matrix size during sampling.
                if hasattr(self.tokenizer, '_bitbpe_enabled') and self.tokenizer._bitbpe_enabled:
                    warnings.warn(
                        "BitBPE model detected with potential generation issues. "
                        "Consider using the baseline BPE model for generation tasks."
                    )
            
            outputs = self.model.generate(**generation_kwargs)
        
        # Decode the outputs
        generated_texts = []
        for output in outputs:
            text = self.tokenizer.decode(output, skip_special_tokens=True)
            generated_texts.append(text)
        
        # Return single string if only one sequence requested
        if num_return_sequences == 1:
            return generated_texts[0]
        return generated_texts
    
    def complete(self, prompt: str, **kwargs) -> str:
        """
        Simple wrapper for generate() that returns only the completion part.
        
        Args:
            prompt: Input text prompt
            **kwargs: Additional arguments passed to generate()
            
        Returns:
            Only the generated completion (without the original prompt)
        """
        full_text = self.generate(prompt, **kwargs)
        # Remove the prompt from the beginning
        completion = full_text[len(prompt):].strip()
        return completion
    
    def chat(self):
        """
        Interactive chat mode for testing the model.
        """
        print("\nEntering interactive chat mode. Type 'quit' to exit.\n")
        
        while True:
            prompt = input("You: ").strip()
            if prompt.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            if not prompt:
                continue
            
            response = self.complete(prompt, max_length=200, temperature=0.8)
            print(f"Model: {response}\n")


def main():
    """Example usage of the ModernGPT2 loader."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Load and test ModernGPT2 model")
    parser.add_argument("--model_path", type=str, default="output/bpe-8k-baseline-small-b200/checkpoint-6641",
                        help="Path to pretrained model")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Device to load model on (cpu, cuda, cuda:0, etc.)")
    parser.add_argument("--interactive", action="store_true",
                        help="Enter interactive chat mode")
    args = parser.parse_args()
    
    # Initialize the loader
    loader = ModernGPT2Loader(args.model_path, args.device)
    
    if args.interactive:
        # Interactive mode
        loader.chat()
    else:
        # Demo mode - show example completions
        print("\n" + "="*50)
        print("Text Completion Examples")
        print("="*50 + "\n")
        
        prompts = [
            "The future of artificial intelligence is",
            "Once upon a time in a distant galaxy",
            "The key to learning programming is",
            "Climate change is one of the most pressing issues",
            "人工知能の未来は",
            "昔々、遠い銀河系で",
            "인공지능의 미래는",
            "기후 변화는 가장 시급한 문제 중 하나",
        ]
        
        for prompt in prompts:
            print(f"Prompt: {prompt}")
            completion = loader.complete(prompt, max_length=100, temperature=0.8)
            print(f"Completion: {completion}\n")
            print("-" * 40 + "\n")


if __name__ == "__main__":
    main()
