<h1>XGBoost</h1>

References
- 

# Overview
XGBoost is an extension of gradient boosted decision trees, with the goal of improving its speed and performance. 

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
- pass it to the `SelectFromModel` class with `threshold` hyperparameter. This threshold is used as cut point for selecting features.
- employ `transform` function to filter out unimportant features from trainset
- train the model with the new set of features

---

# Hyper-parameters

### References: Friedman J.H., 2001, Greedy Function Approximation: A Gradient Boosting Machine [link](http://luthuli.cs.uiuc.edu/~daf/courses/Opt-2017/Papers/2699986.pdf)
Here is a summary of the suggestions for tuning by Freedman:

__Tradeoff between number of trees and learning rate (1)__
- There is a tradeoff between number of trees and learning rate.
- Good practice: start with a  a large value for the number of trees, then tune the shrinkage parameter to achieve the best results. Studies in the paper preferred a shrinkage value of 0.1, a number of trees in the range 100 to 500 and the number of terminal nodes in a tree between 2 and 8.
- Shrinkage parameter between $0<v<1$ controls the learning rate. A very small learning rate ($v \leq 0.1$) leads to a better generalization error.

__Subsampling__
- almost all types of resampling is better than the deterministic approach.
- He suggested a good subsampling fraction is 40% (without replacement), while even lower percentage (20%-30%) at each iteration also improves performance (compare with no sampling approach). For a very large dataset, subsampling smaller than 50% is preferred.
- Also sampling increases the computation speed.
- 


__Terminal nodes__
- terminal nodes value between 3 to 6 perform better than larger values such as 11,21, and 41.

### Book: The Elements of Statistical Learning: Data Mining, Inference, and Prediction
- good value for number of trees at each node = 6 ; and good range is between 4-8. 
- use early stopping, by monitoring validation performance
- emphasizes on the tradeoff between number of trees and learning rate and recommend small value for learning rate (less than 0.1)

### Suggestion by R 
-  number of trees = 100
-  number of leaves = 1
-  min number of samples in tree terminal nodes = 10
-  learning rate = 0.001
-  iterations = 3000 to 10,000 with learning rate btw 0.01 and 0.001

### Suggestion by sklearn
- learning rate = 0.1
- number of trees = 100
- max depth = 3
- min samples split =2
- min samples leaf = 1
- subsamples = 1

### Suggestions by XGBoost
- learning rate = 0.3
- max depth = 6
- subsample = 1

### General suggestions
- run the model by default config
- if overfit, use smaller learning rate (shrinkage)
- if underfit, increase learning rate
- number of trees btw 100 and 1000 depending on the dataset size
- learning rate: $\frac{2~10}{\text{number of trees}}$
- subsample (row sampling): greedsearch in range of [0.5, 0.75, 1.0], another suggestion fix 1.0
- column sampling grid search in range of [0.4, 0.6, 0.8, 1.0], another suggestion grid search in range 0.3 - 0.5
- min leaf weight $\frac{3}{\text{percentage of rate event obs in dataset}}$, another suggestion $\frac{1}{\sqrt{\text{percentage of rate event obs in dataset}}}$
- tree size: grid search [4, 6, 8, 10]
- min split gain (gamma) fixed with at 0.0

---

## Tune hyperparameters
-
