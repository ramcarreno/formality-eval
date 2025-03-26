from abc import ABC, abstractmethod
from typing import Any

import datasets
import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class FormalityModel(ABC):
    """
    An abstract base class for predicting results for all formality detection models.
    Arguments for methods are described only in the ABC. Inheriting classes will include only
    documentation for clarification matters.
    """

    @abstractmethod
    def predict(self, sentence: str) -> dict[str, Any]:
        """
        Returns the probability of being predicted as either formal or informal for a single sentence.

        Args:
            sentence(str):
                The sentence to predict formality for.
        """

    def predict_set(self, test_set: datasets.Dataset)  -> list[int]:
        """
        Perform formality detection predictions on an entire dataset.

        Args:
            test_set(datasets.Dataset):
                Dataset to be evaluated. It must contain a 'text' column.

        Returns:
            list[int]:
                The predicted labels for each text (sentence).
        """


class RuleBased(FormalityModel):
    """
    Although not a traditional model per se, serves as a baseline that checks capitalization and punctuation to
    'predict' formality. Based on the heuristic approach suggested by Dementieva et al. in 'Detecting Text Formality:
    A Study of Text Classification Approaches'. Can be expanded to check for more linguistic rules, wordlists, etc.
    Inherits from the FormalityModel abstract base class.
    """
    def predict(self, sentence: str) -> dict[str, Any]:
        """
        Being a rule-based model, 'prediction' always returns deterministic results.
        """
        if sentence[0].isupper() and sentence.endswith('.'):
            return {"formal": 1, "informal": 0}
        return {"formal": 0, "informal": 1}

    def predict_set(self, test_set: datasets.Dataset) -> list[int]:
        return [0 if sample["text"][0].isupper() and sample["text"].endswith('.') else 1 for sample in test_set]


class EmbeddingsBased(FormalityModel):
    """
    Model based in Word2Vec embeddings.
    Inherits from the FormalityModel abstract base class.
    """

    def predict(self, sentence: str) -> dict[str, Any]:
        pass

    def predict_set(self, test_set: datasets.Dataset):
        pass


class Pretrained(FormalityModel):
    """
    Leverage a pretrained AutoModelForClassification model for the formality detection task.
    Inherits from the FormalityModel abstract base class.
    """
    def __init__(self, model_name: str, batch_size_eval: int):
        """
        Args:
            model_name (str):
                Name of the pretrained model.
            batch_size_eval (int):
                Batch size to use for evaluation.
                It is set to 16 in the __main__ module (recommended if running with CPU).
        """
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
        if len(output) != 2:
            raise ValueError("The predicted output does not match the expected dimensions. "
                             "Only two classes are supported.")
        return {"formal": output[0], "informal": output[1]}

    def predict_set(self, test_set: datasets.Dataset) -> list[int]:
        """
        For pretrained transformer models, tokenization is necessary, along with batch processing to
        avoid memory constraints.
        """
        input_ids = test_set["input_ids"]
        attention_mask = test_set["attention_mask"]
        preds_batch: list[int] = []

        for i in tqdm(range(0, len(input_ids), self.batch_size_eval), desc="Predicting batches..."):
            input_batch = {
                "input_ids": torch.tensor(input_ids[i:i + self.batch_size_eval]),
                "attention_mask": torch.tensor(attention_mask[i:i + self.batch_size_eval])
            }
            output_batch = self.model(**input_batch)
            if output_batch.logits.shape[1] != 2:
                raise ValueError("The predicted output does not match the expected dimensions. "
                                 "Only two classes are supported.")
            preds_batch.extend(output_batch.logits.softmax(dim=1).argmax(dim=1).tolist())  # decode outputs

        return preds_batch

    def predict_set_probabilities(self, test_set: datasets.Dataset) -> list[dict[str, Any]]:
        """
        Instead of direct predictions, returns the probabilities of each class for each sentence in the dataset.
        """
        input_ids = test_set["input_ids"]
        attention_mask = test_set["attention_mask"]
        probs_batch: list[dict[str, Any]] = []

        for i in tqdm(range(0, len(input_ids), self.batch_size_eval), desc="Predicting batches..."):
            input_batch = {
                "input_ids": torch.tensor(input_ids[i:i + self.batch_size_eval]),
                "attention_mask": torch.tensor(attention_mask[i:i + self.batch_size_eval])
            }
            output_batch = self.model(**input_batch)
            probs = output_batch.logits.softmax(dim=1).tolist()
            for j, prob in enumerate(probs):
                probs_batch.append({"text": test_set["text"][i+j], "formal": prob[0], "informal": prob[1]})

        return probs_batch
