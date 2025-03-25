from typing import Any

import datasets
from transformers import PreTrainedTokenizer, AutoTokenizer


class DatasetProcessor:
    """Process a HuggingFace dataset."""
    def __init__(self, data: datasets.DatasetDict, name: str, model_name: str, filter_domain: str=None):
        self.data = data
        self.name = name
        self.filter_domain = filter_domain
        self.tokenizer = None
        if model_name not in ["rule-based", "embeddings"]:
            self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(model_name)
        self.processed_data: dict[str, datasets.Dataset] = self.initialize_processed_splits()

    def initialize_processed_splits(self) -> dict[str, datasets.Dataset]:
        """Apply all necessary processing to all samples and splits"""
        processed_data: dict[str, datasets.Dataset] = {}
        split: str
        for split in ["train", "test"]:
            if split in self.data:
                processed_data[split] = self.data[split].map(
                    self.process_sample,
                    load_from_cache_file=True
                )
                if self.filter_domain is not None:
                    processed_data[split] = processed_data[split].filter(
                        lambda sample: sample['domain'] == self.filter_domain
                    )
        return processed_data

    def process_sample(self, sample: dict[str, Any]) -> dict[str, Any]:
        """
        Processes a single sample for formality detection evaluation, according to dataset needs.
        Default input attr is 'text' and default output attr is 'label'.
        """
        if self.name == "osyvokon/pavlick-formality-scores":
            # normalize input to 'text'
            sample["text"] = sample.pop("sentence")
            # normalize output to 'label' from annotations
            normalized_score = round((sample["avg_score"] - (-3)) / (3 - (-3)) * (1 - (-1)) + (-1), 2)
            sample["norm_score"] = normalized_score
            sample["label"] = 1 if normalized_score <= 0 else 0  # formal: 0, informal: 1
        # elif [...] include HERE processing applicable to other datasets

        # pre-tokenize dataset for pretrained models # TODO: reconsider tokenizing each evaluation batch instead
        if self.tokenizer is not None:
            tokenized = self.tokenizer(
                sample["text"],
                add_special_tokens=True,
                truncation=True,
                padding="max_length",
                max_length=64,
                return_tensors=None
            )
            sample["input_ids"] = tokenized["input_ids"]
            sample["attention_mask"] = tokenized["attention_mask"]

        return sample
