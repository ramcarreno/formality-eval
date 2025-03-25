import datasets
import evaluate

from formality_eval.models import FormalityModel


class Evaluator:
    def __init__(self, model, test_set):
        self.model: FormalityModel = model
        self.dataset: datasets.Dataset = test_set  # use always test set for evaluations
        self.metrics()
        # self.model.predict("This is not working properly.")
        # self.metrics()
        # self.domain_distribution()

    def metrics(self):
        self.predictions = self.model.batch_predict(self.dataset)
        accuracy = evaluate.load("accuracy")
        result = accuracy.compute(predictions=self.predictions, references=self.dataset["label"])
        # TODO: sklearn -> precision_recall_fscore_support()

        return {
            "accuracy": result,
            "precision": 0,
            "recall": 0,
            "f1": 0
        }

    def domain_distribution(self):
        pass