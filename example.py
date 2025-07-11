#!/usr/bin/env python3

import torch
from transformers import AutoTokenizer
from moderngpt2.modeling_moderngpt2 import ModernGPT2LMHeadModel
from typing import Optional, List, Union
import warnings

class ModernGPT2Loader:
    """Wrapper class for loading and using ModernGPT2 models."""
    
    def __init__(self, model_path: str = "output/final/8k-unk/2B", device: str = "cpu"):
        """
        Initialize the model loader.
        
        Args:
            model_path: Path to the pretrained model
            device: Device to load model on ('cpu', 'cuda', 'cuda:0', etc.)
        """
        self.device = torch.device(device)
        self.model_path = model_path
        
        print(f"Loading model from {model_path}...")
        self.model = ModernGPT2LMHeadModel.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        
        print(f"Loading tokenizer from {model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Set pad token if not already set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print(f"Model loaded successfully on {self.device}")
    
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
        # Encode the prompt
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate with the model
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                num_return_sequences=num_return_sequences,
                do_sample=do_sample,
                repetition_penalty=repetition_penalty,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
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
    parser.add_argument("--model_path", type=str, default="output/final/8k-unk/2B",
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
        ]
        
        for prompt in prompts:
            print(f"Prompt: {prompt}")
            completion = loader.complete(prompt, max_length=100, temperature=0.8)
            print(f"Completion: {completion}\n")
            print("-" * 40 + "\n")


if __name__ == "__main__":
    main()
