<h1>XGBoost</h1>

References
- 

# Overview
eXtreme Gradient Boosting (XGBoost) is an extension of gradient boosted decision trees, with the goal of improving its speed and performance. Gradient boosting cosists of three components: 
1. loss function: depending on the problem in hand, differnt loss function is used. However, modern frameworks benefit from an optimized generic loss function.
2. weak learner: the regression trees that are used to produce real values. These real values then added together and the next tree tries to correct the residuals in the prediction. A greedy approach is used for constructing the trees, choosing the best split points based on the purity scocre (e.g. gini).
3. additive model to add weak learners to minimize the loss: A functional gradient descent procedure is used to minimize the loss. The output of a tree is added to the output of previous trees, in an effort to correct or improve the final output of the model. Because of the greedy approach in constructing trees, model has a high tendency for overfitting. There are four main mechanisms implemented to mitigate overfitting
   - tree construction: adding more trees, and put limit on trees' depth (4-8 levels), number of nodes or leaves,  min number of observation per split, and min improvement to loss are all options added to the algorithm to reduce risk of overfitting.
   - shrinkage: The contribution of each tree to this sum can be weighted to slow down the learning by the algorithm. This weight is called shrinkage or learning rate.
   - random sampling: at each iteration a subsample of the training data is drawn at random (without replacement) from the full training dataset. The randomly selected subsample is then used, instead of the full sample, to fit the base learner. A range of subsampling methods are available: subsample rows before creating each tree., subsample columns before creating each tree, or subsample  columns before considering each split.
   - penalized learning: the leaf weight values of trees are regularized with L1 or L2 regularization method.

Some features of XGBoost:
- Algorithm features:
  - handle missing values with sparse aware implementation
  - support the parallelization of tree construction with block construction
  - able to perform constinued training, to handle model update on new data
- system features
  - parallelization of tree construction using multi-thread during training
  - distributed computing for large mdoel
  - out-of-core computing for large dataset
- mdoel features
  - employ learning rate
  - employ subsampling at three levels:  row, column and column per split
  - employ regularization

# Implementation - notes

## Prepare data
XGBoost inherently is an ensemble of regression trees, and requires numerical values as inputs. In case of categorical values, we need to transform them to numerical format. Also, for classification, if the classes are in string format, we need to encode them.

__Classification: output encoding__

In classification task, if the output has string format, we need to employ laber encoding to transform labels to numbers.
```python
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
Y_encoded = label_encoder.fit_transform(Y)
```

__Categorical features__

For the categorical features, we employ transformation to encode categorical to have a numerical representation. However, to avoid interpreting the numerical representation as ordinal, we employ `OneHotEncoder` instead. 
```python
from sklearn.preprocessing import OneHotEncoder
onehot_encoder = OneHotEncoder(sparse=False)
feature = onehot_encoder.fit_transform(feature)
```

__Missing values__
XGBoost automatically learn how to best handles missing values, due to the its design to work with sparse data [ref](https://arxiv.org/abs/1603.02754). However, the expected missing values for XGBosot is zero. A better approach is to specify missing values as `numpy.nan`. Finally, another option is imputation, which can improve or degrate performance dependiing on the imputation method.




---
## Evaluation
__Model evaluation__
- Simplest method is split into train/test. But, the downside is it could result in high variance (the difference between train and test sets could be so high that affect the accuracy difference).  Using a train/test split is good for speed when using a slow algorithm and produces performance estimates with lower bias when using large datasets.
- A better option is k-fold cross-validation, which in many cases considered as a gold-standard for evaluating the performance of a machine learning algorithm (k usually is set to 3, 5, or 10; but the size of dataset plays the ultimate factor in choosing the right k value).

  ```python
  from sklearn.model_selection import KFold
  from sklearn.model_selection import cross_val_score

  # define k value
  kfold = KFold(n_splits=10, random_state=7)
  # evaluate model and return a list of the scores for each model trained on each fold
  results = cross_val_score(model, X, Y, cv=kfold)
  ```
  
  
- Use stratified cross-validation to enforce class distributions when there are a _large number of classes or imbalanced class labels_.

---

__Performance metrics__

XGBoost supports a range of performance metrics including ([Ref.](http://xgboost.readthedocs.io/en/latest/parameter.html):
- rmse for root mean squared error.
- mae for mean absolute error.
- logloss for binary logarithmic loss and mlogloss for multiclass log loss (cross entropy).
- error for classification error.
- auc for area under ROC curve



---
## Others

__Other usefull functions__
- __Plot individual tree__

  Within a trained model, XGBoost Python API has `plot_tree(mdoel, num_trees)` function that plots decision trees (requires `graphviz` library). `num_trees` gets int, representing the decision tree index. 

- __save/load model with pickle__ (`import pickle`):
  - save model: `pickle.dump(model, open("model_checkpoint.dat", "wb"))`
  - load model: `loaded_model = pickle.load(open("model_checkpoint.dat", "rb"))`
- __save/load model with joblib__ (`from sklearn.externals import joblib`):
  - save model: `joblib.dump(model, "model_checkpoint.dat")`
  - load model: `loaded_model = joblib.load("model_checkpoint.dat")`

---

__Feature selection and feature importance scores__

Feature importance score indicates how useful each feature was in the construction of boosted decision trees, within the model.  Importance is calculated for a single decision tree by the amount that each attribute split point improves the performance measure, weighted by the number of observations the node is responsible for.

Using `sklearn.feature_selection.SelectFromModel` we can select most important features to train a model, following these steps:
- pre-train a model
- pass it to the `SelectFromModel` class with `threshold` hyperparameter. This threshold is used as cut point for selecting features.
- employ `transform` function to filter out unimportant features from trainset
- train the model with the new set of features

---

## Hyper-parameters

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
