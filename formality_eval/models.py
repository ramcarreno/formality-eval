from abc import ABC, abstractmethod
from typing import Any

import datasets
import torch
from transformers import PreTrainedTokenizer, AutoTokenizer, AutoModelForSequenceClassification


class FormalityModel(ABC):
    """An abstract base class for receiving predictions for all formality detection models."""

    @abstractmethod
    def predict(self, sentence: str) -> float:
        """Returns a formality prediction label, -1 meaning informal, 1 meaning formal."""
        pass

    def batch_predict(self, test_set: datasets.Dataset):
        pass

class RuleBased(FormalityModel):
    """Simple baseline 'model' that checks capitalization and punctuation."""

    def predict(self, sentence: str) -> float:
        # starting by capital letter and ending punctuated, but not too excessively = formal
        # else informal (Babakov et al.)
        pass

    def batch_predict(self, test_set: datasets.Dataset):
        pass

class EmbeddingsBased(FormalityModel):
    """Model based in Word2Vec embeddings."""

    def predict(self, sentence: str) -> float:
        # starting by capital letter and ending punctuated, but not too excessively = formal
        # else informal
        pass

    def batch_predict(self, test_set: datasets.Dataset):
        pass

class Pretrained(FormalityModel):
    """Leverage a pretrained AutoModelForClassification model for formality."""
    def __init__(self, model_name: str):
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(model_name)

    def predict(self, sentence: str) -> dict[str, Any]:
        input_ids = self.tokenizer(sentence, add_special_tokens=True, return_token_type_ids=True, truncation=True,
                                   padding="max_length", return_tensors="pt")
        outputs = self.model(**input_ids)

        id2formality = {0: "formal", 1: "informal"}

        formality_scores = [
            {id2formality[idx]: score for idx, score in enumerate(text_scores.tolist())}
            for text_scores in outputs.logits.softmax(dim=1)
        ]
        return formality_scores  # TODO: single prediction vs batch prediction

    def batch_predict(self, test_set: datasets.Dataset):
        pass