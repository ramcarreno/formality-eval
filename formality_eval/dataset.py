from typing import Any

import datasets
from transformers import PreTrainedTokenizer, AutoTokenizer


class DatasetProcessor:
    def __init__(self, data: datasets.DatasetDict, name: str, tokenizer: str):
        self.data = data
        self.name = name
        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(tokenizer)
        self.processed_data: dict[str, datasets.Dataset] = self.initialize_processed_splits()

    def initialize_processed_splits(self) -> dict[str, datasets.Dataset]:
        """Apply all necessary processing to all samples and splits"""
        processed_data: dict[str, datasets.Dataset] = {}
        split: str
        for split in ["train", "test"]:
            if split in self.data:
                processed_data[split] = self.data[split].map(
                    self.process_sample,
                    load_from_cache_file=False
                )
        return processed_data

    def process_sample(self, sample: dict[str, Any]):
        """Processes a single sample for formality detection evaluation, according to dataset needs."""
        if self.name == "osyvokon/pavlick-formality-scores":  # processing for dataset
            normalized_score = round((sample["avg_score"] - (-3)) / (3 - (-3)) * (1 - (-1)) + (-1), 2)
            sample["norm_score"] = normalized_score
            sample["label"] = -1 if normalized_score <= 0 else 1
        # [...] processing for other datasets
        return sample


