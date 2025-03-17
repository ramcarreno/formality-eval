import datasets

from formality_eval.models import FormalityModel


class Evaluator:
    def __init__(self, model, test_set):
        self.model: FormalityModel = model  # TODO: use own model classes!
        self.dataset: datasets.Dataset = test_set  # use always test set for evaluations
        self.metrics()
        self.domain_distribution()
        self.pretty_print()

    def metrics(self):  # TODO: own module + class?
        pass

    def domain_distribution(self):
        pass

    def pretty_print(self):
        pass