<h1>XGBoost</h1>

- [Overview](#overview)
- [Hyper-parameters](#parm)


References

- [XG-boost homepage](https://xgboost.readthedocs.io/)
- [XGBoost: A Scalable Tree Boosting System, 2016](http://goo.gl/aFfSef)
- [Lsit of usefull resources](https://github.com/dmlc/xgboost/blob/master/demo/README.md)
- [

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

## Sklearn implenetation
XGBoost comes with different API, including its own API and sklearn API. The sklearnn API benefits from the functions avilable for sklearn, including: `fit` and `predict`, `train_test_split`, `cross_val_score`, `RandomizedSearchCV`, and `GridSearchCV`.

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


---
__Templates - sklearn__

1. classification: Iris dataset - multiclass classification task

   ```python
   import pandas as pd
   import numpy as np
   from sklearn import datasets
   iris = datasets.load_iris() # numpy array - trian and test sets
   df = pd.DataFrame(data= np.c_[iris['data']
      , iris['target']]
      ,columns= iris['feature_names'] + ['target']) # concatenate train and test into single df

   # split to train/test
   from sklearn.model_selection import train_test_split
   X_train, X_test, y_train, y_test = train_test_split(iris['data'], iris['target'], random_state=2)

   # create classification model
   from xgboost import XGBClassifier
   from sklearn.metrics import accuracy_score # other scoring method for classification: AUC

   # initialize model
   xgb = XGBClassifier(
      booster='gbtree' # set base learner
      , objective='multi:softprob' #multi-class classification
      , max_depth=6 # number of branches each tree has
      , learning_rate=0.1 #limit the variance by reducing the weight of each tree to the given percentage
      , n_estimators=100 # number of boosted trees in the model
      , random_state=2
      , n_jobs=-1
      )
   # train the model
   xgb.fit(X_train, y_train)

   # make prediction
   y_pred = xgb.predict(X_test)

   # evaluate model
   score = accuracy_score(y_pred, y_test)
   print(f"Accuracy: {round(score, 2)}")  
   ```
   

3. Regression: diabetes dataset - regression task

   ```python
   # dataset
   X,y = datasets.load_diabetes(return_X_y=True)

   # build model
   from sklearn.model_selection import cross_val_score
   from xgboost import XGBRegressor
   xgb = XGBRegressor(
      booster='gbtree'
      , objective='reg:squarederror'
      , max_depth=6
      , learning_rate=0.1
      , n_estimators=100
      , random_state=2
      , n_jobs=-1
      )

   # train and evaluate model: cross-validation
   scores = cross_val_score(xgb, X, y, scoring='neg_mean_squared_error', cv=5)

   # examine model performance
   rmse = np.sqrt(-scores)
   print('RMSE:', np.round(rmse, 3))
   print('RMSE mean: %0.3f' % (rmse.mean()))
   ```

   Cross-Validation approach
   ```python
   from sklearn.model_selection import cross_val_score

   # model
   model = XGBClassifier(
      booster='gbtree' # base learner: gradient boosted tree
      , objective='binary:logistic'
      , random_state=2
      )

   # train and evaluate
   scores = cross_val_score(model, X, y, cv=5)
   print('Accuracy:', np.round(scores, 2))
   print('Accuracy mean: %0.2f' % (scores.mean()))
   ```

   Stratified cross validation to maintain same percentage of target values in each fold:
   ```python
   from sklearn.model_selection import StratifiedKFold

   kfold = StratifiedKFold(
      n_splits=5
      , shuffle=True
      , random_state=2
      )

   # train and evaluate model
   scores = cross_val_score(model, X, y, cv=kfold)
   print('Accuracy:', np.round(scores, 2))
   print('Accuracy mean: %0.2f' % (scores.mean()))
   ```

   Employ GreedSearch / RandomizedSearchCV for hyperparameter tuning

   ```python
   from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

   # initialize
   xgb = XGBClassifier(booster='gbtree', objective='binary:logistic', random_state=2)

   # train and evaluate
   def train_xgb(random = True):
      if random:
         grid = RandomizedSearchCV(xgb, params, cv=kfold, n_iter=20, n_jobs=-1)
      else:
         grid = GridSearchCV(xgb, params, cv=kfold, n_jobs=-1)
      return grid
      
   # train
   grid.fit(X, y)
   
   # get best values for hyper parameters
   best_params = grid.best_params_
   print("Best params:", best_params)

   # Model score
   best_score = grid.best_score_
   print("Training score: {:.3f}".format(best_score))
   ```

__Template: XGBost API__
- classification - Higgs dataset - binary classification task
  ```python
  # create model - use DMatrix to convert dataframe
  import xgboost as xgb
  xgb_clf = xgb.DMatrix( 
                        X
                       , y
                       , missing=-999.0 # missining values in dataset are defined as -999.0
                       , weight=df['test_Weight']) # for imbalance label, compute weights and assign it
  # define parameters
  param = {
     'objective': 'binary:logitraw' # binary classification
     ,'scale_pos_weight': scale_pos_weight # scale imbalance labels
     ,'eta': 0.1 # learning rate
     ,'max_depth': 6
     ,'eval_metric': 'auc
     }
  # train
  xgb_clf.train(**param)
  ```

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

# Hyper-parameters <a id = 'parm'></a>

__List of important hyperparameters__

![B15551_06_02](https://github.com/user-attachments/assets/3c43485c-c09e-4d0d-a4a2-08631867bcb9)

- `n_estimators`: number of trees trained on the residual in the ensemble. Increase this values for small dataste does not increase performance.
- `learning_rate`: weight applies to the residual for the next tree. Lowering it helps to avoid overfitting. Default is 0.3, while for a large dataset increasing it could improve speed. This hyperparameter and `n_estimator` affect each other. 
- `max_depth`: determines lenght (level) of each tree, and limiting it help to prevent overfitting (by reducing variance). A good range: `[1, 2, 3, 4, 5, 6, 7, 8]`
- `gamma`: provides a threshold that nodes must surpass before making further splits according to the loss function. There is no limit to this hyperparameter, but any value above 10 is very high (defualt is 0).
- `min_child_weight`: the minimum sum of weights required for a node to split into a child. If the sum of the weights is less than the value of min_child_weight, no further splits are made. Value range 1-5 is a good start point.
- `subsample`: represents percentage of rows (training instances) for each boosting round (a good starting point is 0.8).
- `colsample_bytree`: percentage of columns (features) picked randomly for each round (a good starting point 0.7).
- `early_stopping_rounds`: early stopping is not a hyperparameter, but a strategy for optimizing the `n_estimator` parameter. It provides a limit to the number of rounds that iterative machine learning algorithms train on. It stops training after number of consecutive training rounds fail to improve the model.
- `eval_set` and `eval_metric`: can be used as parameter for training (`.fit~) to generate test score for each training round. 
   - `early_stopping_rounds`: an optional parameter to include with `eval_set` and `eval_metric`.

__Strategy and tips for hyperparameter tuning__
- one hyperparameter at a time:
   - start with gridsearch and a range of parameters for `n_estimators: [2, 25, 50, 75, 100]`
   - once get the optimum value for `n_estimators`, add  `max_depth` and use grid search again: `grid_search(params={'max_depth':[1, 2, 3, 4, 5, 6, 7, 8], 'n_estimators':[50]})`
   - once get the optimum values for `n_estimators` and `max_depth`, use a grid search with tight range for both hyperparameters (for example, 2 and 50) to make sure we have overal optimum values: `grid_search(params={'max_depth':[1, 2, 3], 'n_estimators':[2, 50, 100]})`
   - add `learning_rate` with grid search and optimum values for previous hyperparameters: `grid_search(params={'learning_rate':[0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5], 'max_depth':[1], 'n_estimators':[50]})`
   - add `min_child_weight`: `grid_search(params={'min_child_weight':[1, 2, 3, 4, 5], 'max_depth':[1], 'n_estimators':[50], 'learning_rate': [0.2]})`
   - add `subsample`: the same procedure
   - follow the same procedure for the rest of the hyperparameters: `colsample_bylevel`, `colsample_bynode`, `gamma`, ...
   - Employ `RandomizedSearchCV` with tight range for all optimum values dicovered for all hyperparameters to get the best combination, while optimizing for speed.


---

### References: Friedman J.H., 2001, Greedy Function Approximation: A Gradient Boosting Machine [link](http://luthuli.cs.uiuc.edu/~daf/courses/Opt-2017/Papers/2699986.pdf)
Here is a summary of the suggestions for tuning by Freedman:

__Tradeoff between number of trees and learning rate (1)__
- In XGBoost new trees are created to correct the residual errors in the predictions from the existing trees and the weigth factors applies to this correction is called shrinkage factor or learning rate. 
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


### Effect of hyper-parameters
- Number of trees (`n_estimator`) and size of each tree (`max_depth`) are related to each other as we need less larger trees or more shalow trees.
- Picking a smaller learning rate (`learning_rate`) helps reduce the risk of overfitting. Smaller learning rates. generally, requires more trees. While the larger rating rate will result in reaching peak performance with less trees, the model performance is much worse than peaking a small rating rate and more trees.
- Subsampling
   - row subsampling `subsample`: it involves selecting a random sample of the training dataset without replacement. A good range is between 0.2-0.5
   - column subsampling ` colsample bytree`:  it creates a random sample the features prior to creating each decision tree. After around 0.4, the performance plateaus, but it depends heavily on data.
   - column subsampling ` colsample bylevel `: similar to random forest, here subsampling occurs at each split, instead of subsample once for each tree. After value around 0.3, the performance plateaus, but it depends heavily on data.
   - 
---

## Better results by using different base-learners
The base learner is the machine learning model that XGBoost uses to build the first model in its ensemble.

__gbtree__
The default base learner is `gbtree`. 

__gblinear__
A linear gradient boosted model ideal for a case in which there is a linear relashionship btw dep and ind variables. It has linear regularization implemented term, and could be used for both classification and regression tasks. 

Example: XGBRegressor with linear learner
```python
def grid_search(params, reg=XGBRegressor(booster='gblinear')):
    grid_reg = GridSearchCV(reg, params, scoring='neg_mean_squared_error', cv=kfold)
    grid_reg.fit(X, y)
    best_params = grid_reg.best_params_
    print("Best params:", best_params)
    best_score = np.sqrt(-grid_reg.best_score_)
```

__DART__ add dropout techniques to the model, by selecting random sample of previous trees and normalize the leaves by scaling factor computed as ratio of the number of trees dropped. This techniques helps to avoid overfitting. Using this learner requires to set an additional set of hyperparameters to accomodate dropouts. 

Example: XGBRegressor with DART
```python
def regression_model(model):
    scores = cross_val_score(model, X, y, scoring='neg_mean_squared_error', cv=kfold)
    rmse = (-scores)**0.5
    return rmse.mean()


regression_model(XGBRegressor(booster='dart', objective='reg:squarederror'))
```

Example: XGBClassifier with DART
```python
def classification_model(model):
    scores = cross_val_score(model, X_census, y_census, scoring='accuracy', cv=kfold)
    return scores.mean()
classification_model(XGBClassifier(booster='dart'))
```

Hyperparameters
- `sample_type`: Default: "uniform", Range: ["uniform", "weighted"], and Determines how dropped trees are selected
- `normalize_type`: Default: "tree", Range: ["tree", "forest"], Calculates weights of trees in terms of dropped trees
- `rate_drop`: Default: 0.0, Range: [0.0, 1.0], Percentage of trees that are dropped
- `one_drop`: Default: 0, Range: [0, 1], Used to ensure drops
- `skip_drop`: Default: 0.0, Range: [0.0, 1.0], Probability of skipping the dropout
- 

__RandomForest__: There are two strategies to implement random-forest within XGBoost. 
1. use random-forest as base learnerby setting `num_parallel_trees` to a value larger than 1, we will have random-forest as learner in the XGBoost model. NOTE that this method is at experimental stage.
   Hyperparameter: `num_parallel_trees`: Default: 1, Range: [1, inf), Gives number of trees boosted in parallel
2. use XGBoost original randomforests [ref](https://xgboost.readthedocs.io/en/latest/tutorials/rf.html) - as of 2024, the sklearn wrapper for this approach is in experimenal phase.


---

## Tips and tricks for better model





-
