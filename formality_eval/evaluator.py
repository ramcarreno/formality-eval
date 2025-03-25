import datasets
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, classification_report

from formality_eval.models import FormalityModel


class Evaluator:
    def __init__(self, model: FormalityModel, test_set: datasets.Dataset):
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
        """Compute common classification metrics."""
        accuracy = accuracy_score(references, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(references, predictions, average="macro")
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }

    def report(self, references: list[int], predictions: list[int]):
        """Directly print a classification report for the formality detection task."""
        # TODO: write to file?
        print(classification_report(references, predictions, target_names=["formal", "informal"]))
        return

