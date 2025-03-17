from datasets import load_dataset
from transformers import AutoModelForSequenceClassification

from formality_eval import DatasetProcessor, Evaluator

def evaluate():
    model_name: str = "s-nlp/mdeberta-base-formality-ranker"  # TODO: pass as CLI argument
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    hf_dataset_name: str = "osyvokon/pavlick-formality-scores"  # TODO: CLI, accept other dataset formats?
    dataset = DatasetProcessor(
        data=load_dataset(hf_dataset_name),
        name=hf_dataset_name,
        tokenizer=model_name
    )
    # final call
    Evaluator(model=model, test_set=dataset.processed_data["test"])

if __name__ == "__main__":
    evaluate()
