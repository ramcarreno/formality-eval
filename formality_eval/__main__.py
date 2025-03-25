import argparse
from datasets import load_dataset

from formality_eval import DatasetProcessor, Evaluator
from formality_eval import RuleBased, EmbeddingsBased, Pretrained  # Import model classes

def evaluate(model_name: str, dataset_name: str):
    """Evaluate the selected formality detection model on a dataset."""
    # load dataset. only an hf available, set to default
    dataset = DatasetProcessor(
        data=load_dataset(dataset_name),
        name=dataset_name,
        tokenizer_name=model_name
    )

    # load model: our approximations don't take arguments, pretrained use their hf hub name
    if model_name == "rule-based":
        model = RuleBased()
    elif model_name == "embeddings":
        model = EmbeddingsBased()
    else:  # whatever AutoModelForSequenceClassification with formality labels!
        model = Pretrained(model_name=model_name, batch_size_eval=32)
        # options:
        # s-nlp/xlmr_formality_classifier
        # s-nlp/roberta-base-formality-ranker?
        # s-nlp/mdeberta-base-formality-ranker?
        # ...
    # TODO: predict mode (via argument)

    # main call
    Evaluator(model=model, test_set=dataset.processed_data["test"])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate formality models on a dataset.")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Choose the model to use: rule-based, embeddings or a HF pretrained model for formality detection. "
             "In case the latter is chosen, it has to be an AutoModelForSequenceClassification."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="osyvokon/pavlick-formality-scores",
        help="HF dataset. Currently only one is available, so it defaults to it."
    )
    args = parser.parse_args()
    evaluate(model_name=args.model, dataset_name=args.dataset)
