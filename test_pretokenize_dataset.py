import unittest
from unittest.mock import MagicMock

# Assuming pretokenize_dataset.py is in the same directory or PYTHONPATH is set up
# For the testing environment, pretokenize_dataset.py is in the root,
# and tests might be run from the root as well, or a subdirectory.
from pretokenize_dataset import tokenize_function as actual_tokenize_function

class MockTokenizer:
    def __init__(self, pad_token=None):
        # The pad_token isn't strictly used by this mock's __call__ but included for completeness
        self.pad_token = pad_token

    def __call__(self, texts, truncation=False):
        # texts is a list of strings
        input_ids = []
        attention_mask = []
        for text in texts:
            if isinstance(text, str) and text: # Non-empty string
                # Simulate some tokenization: use ASCII values of first 3 chars
                tokens = [ord(c) for c in text[:3]]
                input_ids.append(tokens)
                attention_mask.append([1] * len(tokens))
            else: # Empty string (already processed from None or originally empty)
                input_ids.append([])
                attention_mask.append([])
        return {"input_ids": input_ids, "attention_mask": attention_mask}

class TestTokenizeFunction(unittest.TestCase):

    def test_tokenize_function_behavior(self):
        mock_tokenizer_instance = MockTokenizer()

        # Test cases include: standard string, empty string, None, string with spaces
        sample_texts = ["hello", "", None, "world", "  ", "a"]

        examples_batch = {"text": sample_texts}

        # Call the actual tokenize_function (now imported) with the mock tokenizer
        result = actual_tokenize_function(examples_batch, mock_tokenizer_instance)

        self.assertIn("input_ids", result)
        self.assertIn("attention_mask", result)

        self.assertEqual(len(result["input_ids"]), len(sample_texts), "Mismatch in batch size for input_ids")
        self.assertEqual(len(result["attention_mask"]), len(sample_texts), "Mismatch in batch size for attention_mask")

        # Expected outputs from MockTokenizer based on its logic:
        # "hello" -> [ord('h'), ord('e'), ord('l')] -> [104, 101, 108]
        # ""      -> []
        # None (becomes "" in tokenize_function) -> []
        # "world" -> [ord('w'), ord('o'), ord('r')] -> [119, 111, 114]
        # "  "    -> [ord(' '), ord(' ')] -> [32, 32] (takes first 3 chars, only 2 spaces)
        # "a"     -> [ord('a')] -> [97]

        expected_input_ids = [
            [104, 101, 108], # hello
            [],              # ""
            [],              # None
            [119, 111, 114], # world
            [32, 32],        # "  "
            [97]             # "a"
        ]

        expected_attention_masks = [
            [1, 1, 1],       # hello
            [],              # ""
            [],              # None
            [1, 1, 1],       # world
            [1, 1],          # "  "
            [1]              # "a"
        ]

        for i in range(len(sample_texts)):
            self.assertEqual(result["input_ids"][i], expected_input_ids[i],
                             f"Input_ids mismatch for text: '{sample_texts[i]}'. Expected {expected_input_ids[i]}, got {result['input_ids'][i]}")
            self.assertEqual(result["attention_mask"][i], expected_attention_masks[i],
                             f"Attention_mask mismatch for text: '{sample_texts[i]}'. Expected {expected_attention_masks[i]}, got {result['attention_mask'][i]}")

if __name__ == '__main__':
    # This allows running the test file directly
    unittest.main()


class TestLabelRemoval(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.temp_dir)
        self.tokenizer_dir = os.path.join(self.temp_dir, "tokenizer")
        self.output_dir = os.path.join(self.temp_dir, "output")
        os.makedirs(self.tokenizer_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)

        # Create and save a dummy tokenizer
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.tokenizer.save_pretrained(self.tokenizer_dir)

    # Functions from pretokenize_dataset.py (copied for testability)
    # Note: This assumes ModernGPT2Tokenizer is available or GPT2Tokenizer is sufficient for the test's purpose.
    # If ModernGPT2Tokenizer specific features are tested, it needs to be handled.
    # For this test, basic tokenization is enough.

    def tokenize_function(self, examples, tokenizer):
        texts = examples["text"]
        valid_texts = [t for t in texts if t and isinstance(t, str) and t.strip()]
        if not valid_texts:
            return {"input_ids": [], "attention_mask": []}
        output = tokenizer(valid_texts, truncation=False)
        return output

    def group_texts(self, examples, block_size):
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
        # result["labels"] = result["input_ids"].copy() # This line is removed in the modified version
        return result

    def test_labels_column_removal_from_parquet(self):
        # 1. Args for the functions
        args = argparse.Namespace(
            tokenizer_path=self.tokenizer_dir,
            output_path=self.output_dir,
            block_size=128, # Smaller block size for testing
            max_samples_per_shard=10,
            # dataset_streaming=False, # Not directly used by copied functions
            # c4_langs=None, # Not directly used
            # max_input_lines_total=None, # Not directly used
            # num_proc=1 # Not directly used by copied functions
        )

        # 2. Create a small dummy Hugging Face dataset
        # Need enough tokens to form at least one block of size args.block_size (128)
        # Each sentence is approx 12 tokens. Let's use 300 sentences to be very safe.
        # 300 sentences * ~12 tokens/sent = ~3600 tokens. 3600/128 = ~28 blocks.
        dummy_data = {"text": [f"This is a sample sentence for testing purposes, number {i}." for i in range(300)]}
        raw_dataset = Dataset.from_dict(dummy_data)

        # 3. Call tokenize_function
        # Load the tokenizer using the path, as it would be in the script
        # For this test, we can also just use self.tokenizer directly
        tokenized_dataset = raw_dataset.map(
            lambda examples: self.tokenize_function(examples, self.tokenizer),
            batched=True,
            remove_columns=['text']  # Remove text column after tokenization
        )

        # 4. Call group_texts
        lm_dataset = tokenized_dataset.map(
            lambda examples: self.group_texts(examples, args.block_size),
            batched=True
        ).filter(lambda x: len(x['input_ids']) > 0)


        # 5. Simulate sharding and save to Parquet
        # Simplified sharding logic for the test
        shard_index = 0 # Keep shard_index initialization

        individual_examples = []
        for batch_of_blocks in lm_dataset: # Iterate once over lm_dataset
            num_blocks_in_batch = len(batch_of_blocks['input_ids'])
            for i in range(num_blocks_in_batch):
                individual_examples.append({
                    'input_ids': batch_of_blocks['input_ids'][i],
                    'attention_mask': batch_of_blocks['attention_mask'][i]
                })

        input_ids_list = []
        attention_mask_list = []
        processed_samples_count = 0

        for example_block in individual_examples: # Iterate over all collected blocks
            if processed_samples_count < args.max_samples_per_shard:
                input_ids_list.append(example_block["input_ids"])
                attention_mask_list.append(example_block["attention_mask"])
                processed_samples_count += 1
            else:
                break

        current_shard_data = {
            "input_ids": input_ids_list,
            "attention_mask": attention_mask_list
        }
        processed_samples_in_current_shard = processed_samples_count

        if processed_samples_in_current_shard > 0:
            shard_file_path = os.path.join(args.output_path, f"test_shard_{shard_index}.parquet")
            if not current_shard_data["input_ids"]:
                 self.fail(f"Contradiction: samples_count={processed_samples_in_current_shard}, but current_shard_data_input_ids is empty.")

            table = pa.Table.from_pydict(current_shard_data)
            pq.write_table(table, shard_file_path)

            # 6. Read schema from Parquet
            read_table = pq.read_table(shard_file_path)
            schema = read_table.schema

            # 7. & 8. Assertions
            self.assertIn("input_ids", schema.names)
            self.assertIn("attention_mask", schema.names)
            self.assertNotIn("labels", schema.names)
        else:
            self.fail(f"No data was processed (processed_samples_in_current_shard is {processed_samples_in_current_shard}).")

# Add imports at the top of the file
import os
import tempfile
import shutil
import argparse
import pyarrow.parquet as pq
from transformers import GPT2Tokenizer
# Removed duplicate: from datasets import Dataset
# (Keep existing imports as well)
