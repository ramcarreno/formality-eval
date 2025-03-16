from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from formality_eval.dataset import DatasetProcessor

# tokenizer = AutoTokenizer.from_pretrained("s-nlp/mdeberta-base-formality-ranker")
# model = AutoModelForSequenceClassification.from_pretrained("s-nlp/mdeberta-base-formality-ranker")

dataset = DatasetProcessor(data=load_dataset("osyvokon/pavlick-formality-scores"))