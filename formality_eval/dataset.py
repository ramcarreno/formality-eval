from typing import Any

import datasets
from transformers import PreTrainedTokenizer, AutoTokenizer


class DatasetProcessor:
    """
    Process a HuggingFace dataset.
    """
    def __init__(self, data: datasets.DatasetDict, name: str, model_name: str,
                 formal_label: int, informal_label: int, filter_domain: str=None):
        """
        Args:
            data (datasets.DatasetDict):
                Original dataset, including all its splits.
            name (str):
                Name of the dataset.
            model_name (str):
                Name of the model, used to check whether tokenization is necessary.
            formal_label (int):
                Label that indicates a sentence in the dataset is formal.
            informal_label (int):
                Label that indicates a sentence in the dataset is informal.
            filter_domain (str, Optional):
                Filter dataset by domain, if provided.
        """
        self.data = data
        self.name = name
        self.formal_label = formal_label
        self.informal_label = informal_label
        self.filter_domain = filter_domain
        self.tokenizer = None
        if model_name not in ["rule-based", "embeddings"]:  # only pretrained models need a tokenizer
            self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(model_name)
        self.processed_data: dict[str, datasets.Dataset] = self.initialize_processed_splits()

    def initialize_processed_splits(self) -> dict[str, datasets.Dataset]:
        """
        Apply all necessary processing to all samples and splits, including filtering by domain if domain is
        present in the dataset and specified as a class attribute.

        Returns:
            dict[str, datasets.Dataset]:
                The dataset processed in all its splits.

        Raises:
            KeyError:
                If the dataset wasn't processed for the formality detection task and doesn't contain 'text' and
                'label' features.
        """
        processed_data: dict[str, datasets.Dataset] = {}
        split: str
        for split in ["train", "test"]:
            if split in self.data:
                # process dataset
                processed_data[split] = self.data[split].map(
                    self.process_sample,
                    load_from_cache_file=True
                )
                # check processing followed specified format
                if any(key not in processed_data[split].features for key in ["text", "label"]):
                    raise KeyError(
                        f"Dataset {self.name} doesn't contain either required 'text' or 'label' features."
                        f"Please process your dataset accordingly."
                    )
                # filter dataset by domain if specified and part of the dataset
                if self.filter_domain is not None and "domain" in processed_data[split].features:
                    processed_data[split] = processed_data[split].filter(
                        lambda sample: sample['domain'] == self.filter_domain
                    )
        return processed_data

    def process_sample(self, sample: dict[str, Any]) -> dict[str, Any]:
        """
        Processes a single sample for formality detection evaluation, according to dataset needs.
        Default input attr is 'text' and default output attr is 'label'. Other datasets must be processed here.

        Args:
            sample (dict[str, Any]):
                Each sample of the dataset. Processing will be applied to the entire dataset through the map function.
        Returns:
            dict[str, Any]:
                The processed sample.
        """
        if self.name == "osyvokon/pavlick-formality-scores":
            # normalize input to 'text'
            sample["text"] = sample.pop("sentence")
            # normalize output to 'label' from annotations
            normalized_score = round((sample["avg_score"] - (-3)) / (3 - (-3)) * (1 - (-1)) + (-1), 2)
            sample["norm_score"] = normalized_score
            sample["label"] = self.informal_label if normalized_score <= 0 else self.formal_label
        # add your dataset processing rules HERE!

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
