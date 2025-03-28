import csv
from typing import Any

import datasets
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, classification_report

from formality_eval.models import FormalityModel, Pretrained


class Evaluator:
    def __init__(self, model: FormalityModel, test_set: datasets.Dataset, formal_label: int, informal_label: int):
        """
        Args:
            model (FormalityModel):
                The model to use for predictions.
            test_set (datasets.Dataset):
                The dataset to evaluate on.
            formal_label (int):
                Label that indicates a sentence in the dataset is formal.
            informal_label (int):
                Label that indicates a sentence in the dataset is informal.
        """
        self.model = model
        self.dataset = test_set  # recommended to use always test set for evaluations
        self.formal_label = formal_label
        self.informal_label = informal_label

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
        raise NotImplementedError("Probability prediction is not implemented for this model type.")

    def write_predictions_to_file(self, filename: str):
        """
        Predictions file writer. Current formatting style is limited to that of .csv.

        Args:
            filename (str):
                The name of the output file.
        """
        sentences = self.dataset["text"]
        translated_predictions = ["formal" if label == self.formal_label else "informal" for label in self.predictions]

        # csv writer
        with open(filename, mode="w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["sentence", "prediction"])  # header
            for sentence, pred in zip(sentences, translated_predictions):
                writer.writerow([sentence, pred])