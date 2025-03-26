from typing import Any

import datasets
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, classification_report
from sympy.codegen.ast import Raise

from formality_eval.models import FormalityModel, Pretrained


class Evaluator:
    def __init__(self, model: FormalityModel, test_set: datasets.Dataset):
        """
        Args:
            model (FormalityModel):
                The model to use for predictions.
            test_set (datasets.Dataset):
                The dataset to evaluate on.
        """
        self.model = model
        self.dataset = test_set  # recommended to use always test set for evaluations

        # obtain references and predictions
        self.references = self.dataset["label"]
        self.predictions: list[int] = self.model.predict_set(self.dataset)

        # print evaluation results!
        # self.metrics(self.predictions, self.references)
        self.report(self.references, self.predictions)

    @staticmethod
    def metrics(references: list[int], predictions: list[int]):
        """
        Compute common classification metrics.

        Args:
            references (list[int]):
                Ground truth labels.
            predictions (list[int]):
                Predicted labels.
        """
        accuracy = accuracy_score(references, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(references, predictions, average="macro")
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }

    def report(self, references: list[int], predictions: list[int]):
        """
        Directly print a classification report for the formality detection task.

        Args:
            references (list[int]):
                Ground truth labels.
            predictions (list[int]):
                Predicted labels.
        """
        # TODO: write to file?
        print(classification_report(references, predictions, target_names=["formal", "informal"]))
        return

    def inspect_uncertain_predictions(self, uncertainty_threshold: float) -> list[dict[str, Any]]:
        """
        Filter out predictions where the prediction probability difference is below a certain threshold.
        Only available for pretrained models.

        Args:
            uncertainty_threshold (float):
                The prediction probability difference between classes below dataset sentences are returned.
        """
        if isinstance(self.model, Pretrained):
            prediction_probs = self.model.predict_set_probabilities(self.dataset)
            return [
                pps for pps in prediction_probs
                if abs(pps["formal"] - pps["informal"]) < uncertainty_threshold
            ]
        else:
            raise NotImplementedError("Probability prediction is not implemented for this model type.")
