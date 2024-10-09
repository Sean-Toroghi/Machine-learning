<h1>Utility functions and codes for machine learning</h1> 

This depository consists of two resources: 1. a set of utility codes, and 2. summary and code examples of some of the concepts in machine learning for quick referencing.

## Table of contents

__List of utility codes__
- [SMOTE](https://github.com/Sean-Toroghi/Machine-learning/blob/main/utils/src/smote.py): mitigate imbalanced dataset via oversample-SMOTE
- [preprocessing](https://github.com/Sean-Toroghi/Machine-learning/blob/main/utils/src/preprocessing.py): split univariate seq into samples for supervised learning - For a simple MLP model
- [reduce_memory_usage](https://github.com/Sean-Toroghi/Machine-learning/blob/d3149572d0c1f7b688174038687a5d2f5574f57f/utils/src/reduce_memory_usage.py)
  reduce memory of numerical features in a dataframe. 
- [XGBoost_Train_Chunk.py](https://github.com/Sean-Toroghi/Machine-learning/blob/91058d5c76e795f2a06de91f10d5b5b0729891f4/utils/src/XGBoost_Train_Chunk.py)
  Train XGBoost model on a very large dataset by dividing the dataset into chunks and iteratively going through the data during the training.

__[Summary and code examples](#code)__
- Feature engineering
- Model building
- Other coding 



---

# <a id= 'code'>Summary and code examples</a>

## Feature engineering
If employ tree-based model, employ method such as `.feature_importances_`, could verify which features (created or existed int he dataset) are more significant.


### Categorical features
- _convert to numerical values_ via `pd.get_dummies` or `sklearn.preprocessing.OneHotEncoder`
- _Frequency of each class_: convert categorical columns into their frequencies, which equates to the percentage of times each category appears within the given column.
  ```python
  df['cat_freq'] = df.groupby('categorical_feature')['categorical_feature'].transform('count') 
  df['cat_freq'] = df['cat_freq']/len(df)
  ```
- _Mean encoding (also called target encoding)_: transforms categorical columns into numerical columns based on the mean target variable. To avoid data leakage, we need to apply a regularization technique after the transformation. [Ref.](https://www.kaggle.com/vprokopev/mean-likelihood-encodings-a-comprehensive-study)
  ```python
  from category_encoders.target_encoder import TargetEncoder
  encoder =  TargetEncoder()
  df['categorical_mean_encoded'] = encoder.fit_transform(df['categogircal_feature'], df['target'])
  ```
## Model building

### Ensemble models

One approach to achive a high performance outcome, is to ensemble models. However, it does not mean to include all models, but those that are not correlated. In the case of ensemble models, adding highly correlated models does not benefit the overal performance (as they all predict the same values with small variance). One approach is to employ _majority rules_ for getting the final results.


__Example: ensemble models for a classifciation task__

To perform correlation test on predictions, we first concatenate the results of all models, and then run `.corr` method to obtain correlations. 
```python
def y_pred(model):
  '''return prediction score (accuracy) for the input model'''
  model.fit(X_train, y_train)
  y_pred = model.predict(X_test)
  score = accuracy_score(y_pred, y_test)
  print(score)
  return y_pred

# prepare dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=2)

# get accuracy score for a range of models
y_pred_gbtree = y_pred(XGBClassifier())
y_pred_dart = y_pred(XGBClassifier(booster='dart', one_drop=True))
y_pred_forest = y_pred(RandomForestClassifier())
y_pred_logistic = y_pred(LogisticRegression(max_iter=10000))
y_pred_xgb = y_pred(XGBClassifier(max_depth=2, n_estimators=500, learning_rate=0.1))

# concatenate results
df_pred = pd.DataFrame(data= np.c_[y_pred_gbtree, y_pred_dart, y_pred_forest, y_pred_logistic, y_pred_xgb], columns=['gbtree', 'dart','forest', 'logistic', 'xgb'])

# Compute correlation between results
df_pred.corr()
```
The second step is to implement majority voting (`VotingClassifier` or `VotingRegressor`)
```python
# create models
estimators = []
logistic_model = LogisticRegression(max_iter=10000)
xgb_model = XGBClassifier(max_depth=2, n_estimators=500, learning_rate=0.1)
estimators.append(('xgb', xgb_model))
rf_model = RandomForestClassifier(random_state=2)
estimators.append(('rf', rf_model))

# apply majority voting
ensemble = VotingClassifier(estimators)
# evaluate ensemble model
scores = cross_val_score(ensemble, X, y, cv=kfold)
print(scores.mean())
```

Note that while correlation provides a valuable information, it does not tell the whole story! 

### Stacking models
Stacking combines machine learning models at two different levels: the base level, whose models make predictions on all the data, and the meta level, which takes the predictions of the base models as input and uses them to generate final predictions. In most cases, the meta-model is simple model such as linear regression or logistic-regression. 

Example - stacking approach for a classification task
```python
# create base models
base_models = []
base_models.append(('lr', LogisticRegression()))
base_models.append(('xgb', XGBClassifier()))
base_models.append(('rf', RandomForestClassifier(random_state=2)))

# meta-model
meta_model = LogisticRegression()

# define stacking classifier
clf = StackingClassifier(estimators=base_models, final_estimator=meta_model)

# evalute model
scores = cross_val_score(clf, X, y, cv=kfold)
print(scores.mean())
```














---
---

# coding 

## Loading custom libraries

To avoid the need for restarting kernel every time making change to a custom library to upate the chage apply the following code. The numbers options are as following:
1. `%autoreload 0` (default)- disables the auto-reloading.
2. `%autoreload 1` - only auto-reload modules that were imported using the `%aimport <function>`.
3. `%autoreload 2` - auto-reload all the modules. 
```python
%load_ext autoreload
%autoreload <number 0, 1, or 2>

# in case of 1:
%aimport custom_package
from custom_package import custom_function1, custom_function2, ...

# in case of 2:
from custom_package import custom_function1, custom_function2, ...

```


## print all elements in a list
```python
# option 1
with np.printoptions(threshold=np.inf):
    print(np.array(long_list))

# option 2
import sys
np.set_printoptions(threshold=sys.maxsize)
np.array(cols)

```

## replace header with column names and add the header as a row to the dataset
```python
original_cols = df.columns.tolist()
df.columns = all_cols
df_original_cols = pd.DataFrame([original_cols], columns=all_cols)
df = pd.concat([df, df_original_cols], ignore_index=True)
```

