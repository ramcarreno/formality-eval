# formality_eval 🤵🔎

**formality_eval** is a Python package for _formality detection evaluation_. When running it as a module, it takes a 
formality detection model and a dataset to evaluate the model on, returning prediction probabilities or a classification 
report. 

## Installation

1. Clone the repository and access its root directory
```
git clone https://github.com/ramcarreno/formality-eval.git
cd formality_eval
```
2. Create your virtual environment of choice and install the following dependencies: `datasets`, `torch`, 
`transformers`, `scikit-learn`. If you use `venv`, you can simply do this with:
```
pip install .
```

## Running the package
Before directly running the package with `python -m formality_eval`, the following command line arguments may be added:

- `model`:  This is a **required** argument. It can be any of `rule-based`, `embeddings` or the checkpoint name of a 
HuggingFace pretrained model of the AutoModelForSequenceClassification class.
- `dataset`: A HuggingFace formality dataset. By default it is set to _osyvokon/pavlick-formality-scores_. However, 
given how diversely structured datasets can be, it is suggested that any other used is processed accordingly in the 
_dataset.py_ module as well.
- `predict`: Single sentence prediction mode. Specify here a sentence and it will return the probability it gets
classified as **formal** or **informal**.
- `formal_label` and `informal_label`: In certain cases, one might want to change the labels associated with both
predictions, as some models might use {-1,1}, {0,1}, etc. For formal and informal respectively. By default, 
_formal_label_ is set to 0, and _informal_label_ is set to 1, as it is how the baseline rule-based model encodes 
both.

More specifics about models and datasets are detailed in the document.

### Example run: Classification report
```
python -m formality_eval --model="rule-based"
```
Assuming the default dataset was used, it should return the following report:
```
              precision    recall  f1-score   support

      formal       0.61      0.85      0.71      1017
    informal       0.73      0.43      0.55       983

    accuracy                           0.64      2000
   macro avg       0.67      0.64      0.63      2000
weighted avg       0.67      0.64      0.63      2000
```
### Example run: Predicting formality for a sentence
```
python -m formality_eval --model="s-nlp/xlmr_formality_classifier" --predict="Salutations, fellow reviewer."
```
Should output:
```
{'formal': 0.9984605312347412, 'informal': 0.00153942103497684}
```