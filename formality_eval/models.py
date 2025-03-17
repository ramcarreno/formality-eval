from abc import ABC, abstractmethod


class FormalityModel(ABC):
    """An abstract base class for all formality detection models."""

    @abstractmethod
    def predict(self, text: str) -> float:
        """Returns a formality prediction label, -1 meaning informal, 1 meaning formal."""
        pass


class RuleBased(FormalityModel):
    """Simple baseline 'model' that checks capitalization and punctuation."""

    def predict(self, text: str) -> float:
        # starting by capital letter and ending punctuated, but not too excessively = formal
        # else informal (Babakov et al.)
        pass


class EmbeddingsBased(FormalityModel):
    """Model based in Word2Vec embeddings."""

    def predict(self, text: str) -> float:
        # starting by capital letter and ending punctuated, but not too excessively = formal
        # else informal
        pass


class Pretrained(FormalityModel):
    """Leverage a pretrained AutoModelForClassification model for formality."""
    def __init__(self):
        pass

    def predict(self, text: str) -> float:
        pass