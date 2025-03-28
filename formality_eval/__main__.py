import argparse
from datasets import load_dataset

from formality_eval import DatasetProcessor, Evaluator
from formality_eval import RuleBased, EmbeddingsBased, Pretrained


def evaluate(model_name: str, dataset_name: str, predict_sample: str,
             formal_label: int, informal_label: int, predictions_to_file: str):
    """
    Evaluate the selected formality detection model on a dataset.

    Args:
        model_name (str):
            Refer to --model command line argument help.
        dataset_name (str):
            Refer to --dataset command line argument help.
        predict_sample (str):
            Refer to --predict command line argument help.
        formal_label (int):
            Refer to --formal_label command line argument help.
        informal_label (int):
            Refer to --informal_label command line argument help.
        predictions_to_file (str):
            Refer to --predictions_to_file command line argument help.
    """
    # load dataset. Only HF available, set to default
    dataset = DatasetProcessor(data=load_dataset(dataset_name), name=dataset_name, model_name=model_name,
                               formal_label=formal_label, informal_label=informal_label)

    # load model: approximation models don't take arguments, pretrained transformers use
    # their hf hub name + eval batch size
    if model_name == "rule-based":
        model = RuleBased()
    elif model_name == "embeddings":
        model = EmbeddingsBased(train_set=dataset.processed_data["train"])
    else:  # whatever AutoModelForSequenceClassification for formality detection!
        model = Pretrained(model_name=model_name, batch_size_eval=16)
        # options:
        # s-nlp/xlmr_formality_classifier
        # s-nlp/roberta-base-formality-ranker
        # ...

    # check if "predict" (single sentence prediction) mode has been triggered
    # this mode doesn't perform evaluations on an entire dataset, so it returns when finished
    if predict_sample:
        print(model.predict(predict_sample))
        return

    # main call
    e = Evaluator(model=model, test_set=dataset.processed_data["test"],
              formal_label=formal_label, informal_label=informal_label)
    if predictions_to_file:
        e.write_predictions_to_file(filename=predictions_to_file)


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
        help="HF dataset. Currently only 'osyvokon/pavlick-formality-scores' is available, so it defaults to it."
    )
    parser.add_argument(
        "--predict",
        type=str,
        help="Returns prediction for a single specified sample including associated probabilities."
    )
    parser.add_argument(
        "--formal_label",
        type=int,
        default=0,
        help="The label associated to predicting 'formal'."
    )
    parser.add_argument(
        "--informal_label",
        type=int,
        default=1,
        help="The label associated to predicting 'informal'."
    )
    parser.add_argument(
        "--predictions_to_file",
        type=str,
        help="Name of the file to write model dataset predictions to."
    )
    args = parser.parse_args()
    evaluate(model_name=args.model, dataset_name=args.dataset, predict_sample=args.predict,
             formal_label=args.formal_label, informal_label=args.informal_label,
             predictions_to_file=args.predictions_to_file)
