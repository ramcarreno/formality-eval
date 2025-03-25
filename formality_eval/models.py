from abc import ABC, abstractmethod
from typing import Any, List

import datasets
import torch
from tqdm import tqdm
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
    """Leverage a pretrained AutoModelForClassification model for the formality detection task."""
    def __init__(self, model_name: str, batch_size_eval: int):
        self.batch_size_eval: int = batch_size_eval
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)

    def predict(self, sentence: str) -> dict[str, Any]:
        # TODO: single sample prediction
        return

    # TODO: have fun with this tomorrow, set a batch size, etc.
    def batch_predict(self, test_set: datasets.Dataset) -> list[int]:
        """Perform batch prediction on a dataset containing a 'text' column."""
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
