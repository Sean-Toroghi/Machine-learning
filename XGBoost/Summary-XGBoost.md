<h1>XGBoost</h1>

References
- 

# Overview
XGBoost is an extention of gradient boosted cdecision trees, with the goal of improving its speed and performancce. 

__Model evaluation_
- Generally k-fold cross-validation is the gold-standard for evaluating the performance of a machine learning algorithm on unseen data with k set to 3, 5, or 10.
- Use stratified cross-validation to enforce class distributions when there are a large number of classes or an imbalance in instances for each class.
- Using a train/test split is good for speed when using a slow algorithm and produces performance estimates with lower bias when using large datasets.

---

__Performance metrics__



XGBoost supports a range of performance metrics including ([Ref.](http://xgboost.readthedocs.io/en/latest/parameter.html):
- rmse for root mean squared error.
- mae for mean absolute error.
- logloss for binary logarithmic loss and mlogloss for multiclass log loss (cross entropy).
- error for classification error.
- auc for area under ROC curve



---
- __save model__:
  - regular save: `pickle.dump(model, open("model_checkpoint.dat", "wb"))`
  - seriealized save with joblib: `joblib.dump(model, "model_checkpoint.dat")`
- __load model__:
  - reular load: `loaded_model = pickle.load(open("model_checkpoint.dat", "rb"))`
  - load serialized save model: `loaded_model = joblib.load("model_checkpoint.dat")`

---

__Feature selection with feature importance scores__

Using `sklearn.feature_selection.SelectFromModel` we can select most important features to train a model, following these steps:
- pre-train a model
- pass it to the `SelectFromModel` class with `threshold` hyperparameter. This treshold is used as cut point for selecting features.
- employ `transform` function to filter out unimportant features from trainset
- train the model with the new set of features

---

## Hyper-parameters tuning for XGBoost

### References: Friedman J.H., 2001, Greedy Function Approximation: A Gradient Boosting Machine [link](http://luthuli.cs.uiuc.edu/~daf/courses/Opt-2017/Papers/2699986.pdf)
Here is a summary of the suggestions for tuning by Freedman:

__Tradeoff between numeber of trees and learning rate (1)__
- There is a tradeoff between number of trees and learning rate.
- Good practice: start with a  a large value for the number of trees, then tune the shrinkage parameter to achieve the best results. Studies in the paper preferred a shrinkage value of 0.1, a number of trees in the range 100 to 500 and the number of terminal nodes in a tree between 2 and 8.
- Shrinkage parameter between $0<v<1$ constrols the learning rate. A very small learning rate ($v \leq 0.1$) leads to a better generalization error.

__Subsampling__


__Terminal nodes__


1. Early stopping (

2. 

