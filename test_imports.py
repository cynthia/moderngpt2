import sys
try:
    print("Attempting to import from moderngpt2 (config only):")
    from moderngpt2 import ModernGPT2Config
    print("Successfully imported ModernGPT2Config")
    # from moderngpt2 import ModernGPT2Model  # Skipping model due to missing frameworks
    # print("Successfully imported ModernGPT2Model")
    # from moderngpt2 import ModernGPT2LMHeadModel # Skipping model due to missing frameworks
    # print("Successfully imported ModernGPT2LMHeadModel")
    # from moderngpt2 import ModernGPT2Tokenizer # Skipping tokenizer due to missing frameworks issue
    # print("Successfully imported ModernGPT2Tokenizer")
    # from moderngpt2 import ModernGPT2TokenizerFast # Skipping tokenizer due to missing frameworks issue
    # print("Successfully imported ModernGPT2TokenizerFast")
    # from moderngpt2 import FlaxModernGPT2Model # Skipping model due to missing frameworks
    # print("Successfully imported FlaxModernGPT2Model")
    # from moderngpt2 import TFModernGPT2Model # Skipping model due to missing frameworks
    # print("Successfully imported TFModernGPT2Model")

    # Test a configuration instantiation
    config = ModernGPT2Config()
    print(f"Successfully instantiated ModernGPT2Config: {config.model_type}")

    print("\nConfig import and instantiation successful!")
    sys.exit(0)
except ImportError as e:
    print(f"ImportError occurred: {e}")
    sys.exit(1)
except Exception as e:
    print(f"An unexpected error occurred: {e}")
    sys.exit(1)
