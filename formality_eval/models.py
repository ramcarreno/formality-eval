from abc import ABC, abstractmethod
from typing import Any

import datasets
import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class FormalityModel(ABC):
    """An abstract base class for predicting results for all formality detection models."""

    @abstractmethod
    def predict(self, sentence: str) -> dict[str, Any]:
        """
        Returns the probability of being predicted as either formal or informal for a single sentence.

        Args:
            sentence(str):
                The sentence to predict formality for.
        """

    def predict_set(self, test_set: datasets.Dataset):
        """
        Perform formality detection predictions on an entire dataset.

        Args:
            test_set(datasets.Dataset):
                Dataset to be evaluated. It must contain a 'text' column.
        """


class RuleBased(FormalityModel):
    """
    Although not a traditional model per se, serves as a baseline that checks capitalization and punctuation to
    'predict' formality. Based on the heuristic approach suggested by Babakov et al. in 'Detecting Text Formality:
    A Study of Text Classification Approaches'. Can be expanded to check for more linguistic rules, wordlists, etc.
    """
    def predict(self, sentence: str) -> dict[str, Any]:
        """Being a rule-based model, 'prediction' always returns deterministic results."""
        if sentence[0].isupper() and sentence.endswith('.'):
            return {"formal": 1, "informal": 0}
        return {"formal": 0, "informal": 1}

    def predict_set(self, test_set: datasets.Dataset) -> list[int]:
        return [1 if sample["text"][0].isupper() and sample["text"].endswith('.') else 0 for sample in test_set]


class EmbeddingsBased(FormalityModel):
    """Model based in Word2Vec embeddings."""

    def predict(self, sentence: str) -> dict[str, Any]:
        pass

    def predict_set(self, test_set: datasets.Dataset):
        pass


class Pretrained(FormalityModel):
    """Leverage a pretrained AutoModelForClassification model for the formality detection task."""
    def __init__(self, model_name: str, batch_size_eval: int):
        self.batch_size_eval: int = batch_size_eval
        self.model_name = model_name
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def predict(self, sentence: str) -> dict[str, Any]:
        tokenized_sentence = self.tokenizer(
            sentence,
            add_special_tokens=True,
            return_tensors="pt"
        )
        output = self.model(**tokenized_sentence).logits.softmax(dim=-1)[0].tolist()
        return {"formal": output[0], "informal": output[1]}

    def predict_set(self, test_set: datasets.Dataset) -> list[int]:
        """For pretrained transformer models, tokenization is necessary, along with batch processing to
        avoid memory constraints."""
        input_ids = test_set["input_ids"]
        attention_mask = test_set["attention_mask"]
        preds_batch: list[int] = []

        for i in tqdm(range(0, len(input_ids), self.batch_size_eval), desc="Evaluating batches..."):
            input_batch = {
                "input_ids": torch.tensor(input_ids[i:i + self.batch_size_eval]),
                "attention_mask": torch.tensor(attention_mask[i:i + self.batch_size_eval])
            }
            output_batch = self.model(**input_batch)
            preds_batch.extend(output_batch.logits.softmax(dim=1).argmax(dim=1).tolist())  # decode outputs

        return preds_batch
