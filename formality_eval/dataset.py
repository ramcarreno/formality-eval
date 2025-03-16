from typing import Any

import datasets


class DatasetProcessor:
    def __init__(self, data: datasets.DatasetDict):
        self.data = data
        self.processed_data = self.initialize_processed_splits()

    def initialize_processed_splits(self) -> dict[str, datasets.Dataset]:
        processed_data: dict[str, datasets.Dataset] = {}
        split: str
        for split in ["train", "test"]:
            if split in self.data:
                processed_data[split] = self.data[split].map(
                    self.process_sample,
                    fn_kwargs={"normalize_scores": True},  # TODO: make it a class attribute
                )
        return processed_data

    @staticmethod
    def process_sample(sample: dict[str, Any], normalize_scores: bool):
        # TODO: from score to tag
        if normalize_scores:
            normalized_score = round((sample["avg_score"] - (-3)) / (3 - (-3)) * (1 - (-1)) + (-1), 2)
            sample["norm_score"] = normalized_score
        return sample


