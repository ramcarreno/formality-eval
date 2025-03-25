import argparse
from datasets import load_dataset

from formality_eval import DatasetProcessor, Evaluator
from formality_eval import RuleBased, EmbeddingsBased, Pretrained


def evaluate(model_name: str, dataset_name: str, predict_sample: str):
    """Evaluate the selected formality detection model on a dataset."""
    # load dataset. Only HF available, set to default
    dataset = DatasetProcessor(data=load_dataset(dataset_name), name=dataset_name, model_name=model_name)

    # load model: approximation models don't take arguments, pretrained transformers use
    # their hf hub name + eval batch size
    if model_name == "rule-based":
        model = RuleBased()
    elif model_name == "embeddings":
        model = EmbeddingsBased()
    else:  # whatever AutoModelForSequenceClassification for formality detection!
        model = Pretrained(model_name=model_name, batch_size_eval=16)
        # options:
        # s-nlp/xlmr_formality_classifier
        # s-nlp/roberta-base-formality-ranker
        # s-nlp/mdeberta-base-formality-ranker?
        # ...

    # check if "predict" (single sentence prediction) mode has been triggered
    # this mode doesn't perform evaluations on an entire dataset
    if predict_sample:
        print(model.predict(predict_sample))
        return

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
    parser.add_argument(
        "--predict",
        type=str,
        help="Returns prediction for a single specified sample including associated probabilities."
    )
    args = parser.parse_args()
    evaluate(model_name=args.model, dataset_name=args.dataset, predict_sample=args.predict)
