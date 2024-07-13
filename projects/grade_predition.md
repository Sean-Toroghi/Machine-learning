<h1>Predict grades</h1>

# Overview
__Goal__: to predict student grades to target services aimed at bridging the tech skills gap.

__Steps__
- loading data, cleaning/preprocessing data (missing value/imputation, one-hot-encoding for categorical features, convert to compress sparse matrix)
- develop a customized transformers to automate preprocessing steps
- seperate data into X,y; split into train/test;
- develop preprocessing pipeline: put previous two steps into `sklearn.pipeline import Pipeline` method
- build model and add it to the pipeline:
  - build a base model; employ `Kfold` method to build a cross-val function
  - fine-tune to improve performance via `GridSearchCV` or `RandomizedSearch`
  - test the model performance against test test
  - final touch: choose one of the following:
    - Return to hyperparameter fine-tuning.
    - Keep the model as is.
    - Make a quick adjustment based on hyperparameter knowledge.
  - add the finalized model to the pipeline
- 
