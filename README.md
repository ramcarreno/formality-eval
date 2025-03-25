**formality_eval** is a Python package for _formality detection evaluation_. When running it as a module, it takes a 
formality detection model and a dataset to evaluate the model on, returning prediction probabilities or a classification 
report. 

## Directly running the package
In order to directly run the package, the following command line arguments may be used:

- `model`:  This is a required argument. Any of `rule-based`, `embeddings` or the checkpoint name of a HuggingFace 
pretrained model of the AutoModelForSequenceClassification class. By default, the evaluator uses `0` as the reference 
label for the **formal** class and `1` for **informal**. Certain pretrained models such as 
_s-nlp/roberta-base-formality-ranker_ use the opposite labeling. Labels can be changed in the `dataset.py` module.
- `dataset`: A HuggingFace formality dataset. By default it is set to `osyvokon/pavlick-formality-scores`. However, 
given how diversely structured datasets can be, it is suggested that any other used is processed accordingly in the 
`dataset.py` module as well.
- `predict`: Single sentence prediction mode. Specify here a sentence and it will return the probability it gets
classified as **formal** or **informal**.

More specifics about models and datasets are detailed in the document.

### Example run: Classification report
```
python -m formality_eval --model="s-nlp/xlmr_formality_classifier"
```
It should return the following report:
```
              precision    recall  f1-score   support

      formal       0.61      0.98      0.75      1017
    informal       0.94      0.35      0.51       983

    accuracy                           0.67      2000
   macro avg       0.77      0.66      0.63      2000
weighted avg       0.77      0.67      0.63      2000
```
### Example run: Predicting formality for a sentence
```
python -m formality_eval --model="s-nlp/xlmr_formality_classifier" --predict="Salutations, fellow reviewer."
```
Should output:
```
{'formal': 0.9984605312347412, 'informal': 0.00153942103497684}
```

## Other features

- Getting metrics directly
- Filtering dataset by domain
- Change evaluation batch size (for pretrained transformer models)
- [...]