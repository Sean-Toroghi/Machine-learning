# LightGBM

References
- [Machine Learning with LightGBM and Python by Andrich van Wyk](https://learning.oreilly.com/library/view/machine-learning-with/9781800564749/B16690_01.xhtml)

---
## Overview

Data preprocessing pipeline for a machine learning task usually follows these steps:


__Overfitting__ is a phenomenon in the ML field, occurs when a model memorize the data and fit the noise, leading to lose the generalization ability. Overfitting stem from one or combination of the following factors:
- An overly complex model: A model that is too complex for the amount of data we have utilizes additional complexity to memorize the noise in the data, leading to overfitting
- Insufficient data: If we don’t have enough training data for the model we use, it’s similar to an overly complex model, which overfits the data
- Too many features: A dataset with too many features likely contains irrelevant (noisy) features that reduce the model’s generalization
- Overtraining: Training the model for too long allows it to memorize the noise in the dataset

__Avoid overfitting__: the following actions (single or combination) are used to avoid overfitting:
- Early stopping: We can stop training when we see the validation error beginning to increase.
- Simplifying the model: A less complex model with fewer parameters would be incapable of learning the noise in the training data, thereby generalizing better.
- Get more data: Either collecting more data or augmenting data is an effective method for preventing overfitting by giving the model a better chance to learn the signal in the data instead of the noise in a smaller dataset.
- Feature selection and dimensionality reduction: As some features might be irrelevant to the problem being solved, we can discard features we think are redundant or use techniques such as Principal Component Analysis to reduce the dimensionality (features).
- Adding regularization: Smaller parameter values typically lead to better generalization, depending on the model (a neural network is an example of such a model). Regularization adds a penalty term to the objective function to discourage large parameter values. By driving the parameters to smaller (or zero) values, they contribute less to the prediction, effectively simplifying the model.
- Ensemble methods: Combining the prediction from multiple, weaker models can lead to better generalization while also improving performance.

---

__Model performance metrics__ [Reference](https://scikit-learn.org/stable/modules/model_evaluation.html)

Model performance refers to the ability of a machine learning model to make accurate predictions or generate meaningful outputs based on the given inputs. It basically shows how good a model learned the underlying pattern of existing data, and generalizes to new data.

__1- cleasification__
- four indicators (TP,TN, FP, FN)
- accuracy: number of correct predictions divided by the total number of predictions.
- precision = $\frac{TP}{\text{all positive predictions}}$ indicates how precise the model is in predicting positives.
- recall  = $\frac{TP}{\text{all positive instances}} measures how effectively the model finds (or recalls) all true positive cases.
- F1-score is the harmonic mean between precision and recall.

__2- regression metrics__
- mean square error: average of the squared differences between predicted and actual values. While it is differentiable (can be used in gradient based learning), it penalizes large errors more heavily that small errors (due to the squaring the difference).
- mean absolute error 
- average of the absolute differences between predicted and actual values. It is more robust against the size of errors and less sensitive to outliers. Unfortunately, it is not differentiable!

  
---
## Tree-based models

__Entropy and information gain__

Entropy is a way to measure the disorder or randomness of a system, and measures how surprising the result of a specific input or event might be.  

Information gain is the amount of information gained when modifying or observing the underlying data, and involves reducing entropy from before the observation.

Tree-based models employ different approach to measure information gain or entropy, among which are
- gini index
- log-loss or entropy

The regression task requires additional metric to determine splits at each node, among which are ([ref.](https://scikit-learn.org/stable/modules/tree.html#regression-criteria)):
- MSE or MAE
- half Poisson distance


__Advantages of tree-based models__
- they could use both numerical and categorical features
- less sensitive to data range and size, leads to reducing data preparation effort
- the result is interpretable

__Disadvantages of tree-based models__
- prone to overfitting
- perform poor at extrapolation tasks
- perform poor when trained on unbalanced data. The high-frequency classes will dominate the prediction.

__Mitigate overfitting issue__: there are several strategies implemented tree-based models to overcome overfitting issue, among which are:
- pruning: removing branches that do not contribute much information gain, leading to reduce model complexity.
- control max depth: limit the depth of a tree helps to reduce model complexity.
- control max number of leaf nodes: helps to avoid creation of over-specific branches, leading to less complex model.
- control min number of samples per leaf: stops splitting, if the number of samples reaches a limit, avoids overly specific leaf nodes, and leads to less complex model
- enemble

  

--- 

__Decision tree hyper-parameters__ [regression ref.](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html#sklearn-tree-decisiontreeclassifier),  [classification ref.](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html#sklearn.tree.DecisionTreeRegressor)

Decition tree model, as a tree-based model, contains a range of hyperparameters, among which are
- max_depth: The maximum depth the tree is allowed to reach. Deeper trees allow more splits, resulting in more complex trees and overfitting.
- min_samples_split: The minimum number of samples required to split a node. Nodes containing only a few samples overfit the data, whereas having a larger minimum improves generalization.
- min_samples_leaf: The minimum number of samples allowed in leaf nodes. Like the minimum samples in a split, increasing the value leads to less complex trees, reducing overfitting.
- max_leaf_nodes: The maximum number of lead nodes to allow. Fewer leaf nodes reduce the tree size and, therefore, the complexity, which may improve generalization.
- max_features: The maximum features to consider when determining a split. Discarding some features reduces noise in the data, which improves overfitting. Features are chosen at random.
- criterion: The impurity measure to use when determining a split, either gini or entropy/log_loss.

---
## Ensemble methods

To bring diversity to the ensemble, we can train models on subset of samples, subset of features, different models, different set of hyper-parameters, or differentiate by diversify a part of model (such as embedding, or random initialization of parameters). Some of the most significant ensemble approaches are:
- bagging: train on subset of samples and features
- boosting: interactively train models on the error or the previous models
- stacking: train multiple based models, and higher-order models (meta-models) are then trained to learn from base model predictions, and make final prediction
- blending: meta-models are first trained on prediction made by the base-models on a hold-out set (a part of the training data the base learners were not trained on)

--- 
### Bagging methods

Some of the available bagging models are random forest and extra tree:

__Random forest__

An ensemble tree-based model, based on bagging concept. Some of the hyper-parameters of random-forest are [ref.](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html):
- n_estimators: Controls the number of trees in the forest. Generally, more trees are better. However, a point of diminishing returns is often reached.
- max_features: Determines the maximum number of features to be used as a subset when splitting a node. Setting max_features=1.0 allows all features to be used in the random selection.
- bootstrap determines whether bagging is used. All trees use the entire training set if bootstrap is set to False.


__Extra tree__

This method applies randomness inside the model. Some nodes split samples randomly insteado using the Gini index or information gain. 


### Boosting method


__Gradient boost decision tree__

GBDT (also called multiple additive regression trees (MART)) method sequentially train models, each learns from the mistakes of the previous models. Each model uses the entire dataset. Some of the hyper-parameters for gradient boosting method are as follow ([ref](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html)):
- n_estimators: Controls the number of trees in the ensemble. Generally, more trees are better. However, a point of diminishing returns is often reached, and overfitting occurs when there are too many trees.
- learning_rate: Controls the contribution of each tree to the ensemble. Lower learning rates lead to longer training times and may require more trees to be built (larger values for n_estimators). Setting learning_rate to a very large value may cause the optimization to miss optimum points and must be combined with fewer trees. Then, the DART algorithm is to apply additional scaling of the contribution of the new tree.

__DART__

DART is an extension of GBDT, which employs dropouts to avoid overfitting. First, the next decision tree is built from a random subset of the previous tree, with probability $p$ that indicates probability of previous tree being included. DART is implemented as a part of LightGBM.

---
---

# LightGBM

---

## Overview

LightGBM is a gradient-boosting framework for tree-based ensemble method, with focus on efficiency (swpace and time), while improving accuracy. Applications with high-dim and large data size is where LightGBM shines. It supports both regression and classification (binary or multiclass) tasks, and ranking via LambdaRank. LightGBM employs several techniques and provides customization through hyper-parameter tuning, including DART, bagging, continuous training, early stopping, and several other options.

__Optimization__

To improve model efficiency, Lightbgm benefits from __histogram sorting__ method:
- Reduce computation complexity via _histogram sorting_: the most computational intense task of a GBDT is training the regression tree for each iteration, where finding the optimal split is very expensive. Algorithm requires to sort the samples (either at prior to splitting, or at the time of splitting). By creating feature histograms, the time-complexity for building a decision node reduce from $O(n)$ (pre-sorting approach) to $O(b)$, where $b$ is number of bins.
- Employ _histogram subtraction_ for building the histograms for the leaves. This approach subtracts the leaf’s neighbor’s histogram from the parent’s histogram, instead of calculating the histogram for each leaf.
- Employ histogram to reduce space complexity (memory cost). Any sorting method requires memory allocation, while histogram sorting does mot. Also, a more efficient data type is required to store bins (number of bins is much smaller than the size of dataset).

LightGBM employs __exclusive feature bundling (EFB)__ to optimize working with sparse data.

LightGBM employs __gradient-based one-side sampling (GOOS)__, which discard samples that do not contribute significantly to the training process, leading to reduse size of the training data.

__Tree growth__

LightGBM employs leaf-wise approach to grow the tree, which selects an existing leaf with the most significant change in the loss of the tree and builds the tree from there.

__L1 and L2 regularization__

LightGBM supports both L1 and L2 regularization:
- L1 regularization has the effect of driving leaf scores to zero by penalizing leaves with large absolute outputs.
- L2 regularization has an outsized effect on outliers’ leaves due to taking the square of the output.
- Both regularization also prevents the tree to get too large.

---
### LightGBM hyper-parameters

While LightGBM has many hyper-parameters, they can be divided into core framework, accuracy related, and learning control (avoid overfitting) parameters:

__Core framework parameters__:

- objective: LightGBM supports the following optimization objectives, among others—regression (including regression applications with other loss functions such as Huber and Fair), binary (classification), multiclass (classification), cross-entropy, and lambdarank for ranking problems.
- boosting: The boosting parameter controls the boosting type. By default, this is set to gbdt, the standard GBDT algorithm. The other options are dart and rf for random forests. The random forest
- mode does not perform boosting but instead builds a random forest.
- num_iterations (or n_estimators): Controls the number of boosting iterations and, therefore, the number of trees built.
- num_leaves: Controls the maximum number of leaves in a single tree.
- learning_rate: Controls the learning, or shrinkage rate, which is the contribution of each tree to the overall prediction.

__Accuracy parameters__:

- boosting: Use dart, which has been shown to outperform standard GBDTs.
- learning_rate: The learning rate must be tuned alongside num_iterations for better accuracy. A small learning rate with a large value for num_iterations leads to better accuracy at the expense of optimization speed.
- num_leaves: A larger number of leaves improves accuracy but may lead to overfitting.
- max_bin: The maximum number of bins in which features are bucketed when constructing histograms. A larger max_bin size slows the training and uses more memory but may improve accuracy.

__Learning control parameters__:
- bagging_fraction and bagging_freq: Setting both parameters enables feature bagging. Bagging may be used in addition to boosting and doesn’t force the use of a random forest. Enabling bagging reduces overfitting.
- early_stopping_round: Enables early stopping and controls the number of iterations used to determine whether training should be stopped. Training is stopped if no improvement is made to any metric in the iterations set by early_stopping_round.
- min_data_in_leaf: The minimum samples allowed in a leaf. Larger values reduce overfitting.
- min_gain_to_split: The minimum amount of information gain required to perform a split. Higher values reduce overfitting.
- reg_alpha: Controls L1 regularization. Higher values reduce overfitting.
- reg_lambda: Controls L2 regularization. Higher values reduce overfitting.
- max_depth: Controls the maximum depth of individual trees. Shallower trees reduce overfitting.
- max_drop: Controls the maximum number of dropped trees when using the DART algorithm (is only used when boosting is set to dart). A larger value reduces overfitting.
- extra_trees: Enables the Extremely Randomized Trees (ExtraTrees) algorithm. LightGBM then chooses a split threshold at random for each feature. Enabling Extra-Trees can reduce overfitting. The parameter can be used in conjunction with any boosting mode.

---

### Limitation of LightGBM
- sensitive to overfitting
- important role of fine-tuning
- requires feature engineering
- cannot directly use sequential data. Requires feature engineering (create lagged features) prior to training
- may not able to find complex feature interaction and non-linearity

---

### LightGBM components



---
__LightGBM API__

LightGBM has Python API as well as sklearn implementation. 

__Python API__
- __Dataset__ wrapper class: supports numpy array and pandas dataframe, in addition to a Path to common dataset format such as CSV, TSV, LIBSVM text file, or LightGBM Dataset binary file.
- __LightGBM supports callbacks__. A callback is a hook into the training process that is executed each boosting iteration.
- __Prediction__: The LightGBM predict function (for classification task) outputs an array of activations, one for each class.


__Sklearn API__: 

sklearn implementation does not require `Dataset` wrapper. The sklearn contains four models: 
- LGBMModel,
- LGBMClassifier,
- LGBMRegressor, and
- LGBMRanker


---
---

### LightGBM vs XGBoost vs Deep Learning

#### XGBoost 

[ref](https://xgboost.readthedocs.io/en/stable/parameter.html)

XGBoost is another ensemble tree-based model, with the following features:
- Regularization: XGBoost incorporates both L1 and L2 regularization to avoid overfitting
- Sparsity awareness: XGBoost efficiently handles sparse data and missing values, automatically learning the best imputation strategy during training
- Parallelization: The library employs parallel and distributed computing techniques to train multiple trees simultaneously, significantly reducing training time
- Early stopping: XGBoost provides an option to halt the training process if there is no significant improvement in the model’s performance, improving performance and preventing overfitting
- Cross-platform compatibility: XGBoost is available for many programming languages, including Python, R, Java, and Scala, making it accessible to a diverse user base

#### XGBoost vs LightGBM

Similarities:
- Both libraries implement GBDTs and DART and support building random forests.
- Both have similar techniques to avoid overfitting and handle missing values and sparse data automatically.

Differences:
- Tree-growing strategy: XGBoost employs a level-wise tree growth approach, where trees are built level by level, while LightGBM uses a leaf-wise tree growth strategy that focuses on growing the tree by choosing the leaf with the highest delta loss. This makes LightGBM faster.
- Speed and scalability: LightGBM is designed to be more efficient regarding memory usage and computation time, making it a better choice for large-scale datasets or when training time is critical. However, this speed advantage sometimes results in higher variance in model predictions.
- Handling categorical features: LightGBM has built-in support for categorical features, meaning it can handle them directly without needing one-hot encoding or other preprocessing techniques. XGBoost, on the other hand, requires preprocessing for categorical features before feeding them into the model.
- Early stopping: XGBoost provides an option to halt the training process if there is no significant improvement in the model’s performance. LightGBM does not have this feature built in, although it can be implemented manually using callbacks.

---

#### Deep learning - TabTransformer

Although there are several deep learning architecture, here I provide one of the more recent architecture for illustration: TabTransformer. This method is designed to handle tabular dataset, with mix feature type: numerical and categorical. Before feeding it to the model, it normalizes the numerical features, and employ embedding/ transformer-based model for categorical features. Then it concatenates the two, and feed them into a MLP model


---
---

# Machine learning with LightGBM

__Validate model performance__

- __Cross validation__: an alternative to split data into train/val/test set, is cross-validation, in which the dataset splitting multiple times and train the model multiple times, once for each split.
- __Stratified k-fold validation__: preserves the percentage of samples for each class when creating folds. This way, all folds will have the same distribution of classes as the original dataset.

__Parameter optimization (parameter tuning)__

- Naïve strategy: try an extensive range of values for a parameter, find the best value, and then repeat the process for the following parameter. However, because several hyper-parameters are co-dependent, any change to a hyperparameter could change the optimal value for other parameters.
- __Grid search__ `GridSearchCV`: An exhaustive search over all parameter, training and validating the model on each possible combination of parameters.

--- 
## Parameter optimization
All ensemble tree-based models require to have a rigorous fine-tuning to find the optimal hyperparameters. The hyperparameters significantly impact the algorithm’s performance and generalization capability. The optimal parameters are also specific to the model used and the learning problem being solved. As the result, the first step after building a LightGBM model is to optimize its hyperparameters. What makes this a complex task is threefolds: 
- cost associate with performing try and error approach is high. For each setup, we need to run the model to evalute the effect of the chosen hyper-parameters on its performance.
- high-dimension search space make it impossible to test every values for hyperparameters, provided each can take a range of vlaues.
- parameter interaction with each other, which makes it impossible to test the effect of a hyper-parameter in an isolated environment.

### Visualization aid to investigate interaction between hyperparametr
__Parallel coordiation plot__ is a visualizaton aid that can be used to investigate effect of hyper-parameters on each other and on the model performance. It is very useful to visualize the result of hyper-parameter optimization task and pinpoint which hyperparameter values or combinations are most conducive to optimal model performance.

Each dimension (in this context, a hyperparameter) is portrayed as a vertical axis arranged in parallel. The range of each axis mirrors the range of values that the hyperparameter can assume. Every individual configuration of hyperparameters is depicted as a line crossing all these axes, with the intersection point on each axis indicating the value of that hyperparameter for the given configuration.


There are three approaches for hyper-parameter optimization: manual, bruteforce and optimized approach. 
-  With manual optimization, a human practitioner selects parameters based on intuitive understanding and experience. A model is trained with these parameters, and the process is repeated until satisfactory parameters are found. Manual optimization is simple to implement but is very time-consuming due to the human-in-the-loop nature of the process. Human intuition is also fallible, and good parameter combinations can easily be missed.
-  The brute-force approach exhaustively tested each possible combination to find the optimal values. _Grid search_ can be used to perform this approach. With grid search a range of values for each hyperparameter is given to the algorithm and it checks every combination. The downside for this appraoch is its cost.
-  Auomated and optimized approach employs a set of algorithm to find the best combination of hyper-parameters efficiently. 

Several frameworks are proposed for this purpose: grid search, SHERPA, Hyperopt, Talos, and other.

### Optuna [Link](https://optuna.org/)
 Optuna is a hyperparameter optimizaton framework of the optimization algorithm that finds the best set of hyper-parameters for a machine learning model. Some of the Optuna features are:
 - efficient optimization alogrithm to search hyperparameter space
 - pruning strategy that elimiates inefficient hyperparameters
 - capable of getting parameter type as input (int, float, or categorical)
 - visualization feature for the results

Optuna provides optimization and pruning in efficient manner [ref](https://github.com/optuna/optuna/wiki/Benchmarks-with-Kurobako).


__Optimization__

The optimization part of optuna is done via a range of algorithms among which are: tree-structured Parzen estimator (TPE), and covariance matrix adaptation evolution strategy (CMA-ES) algorithm.

- TPE: it uses kernel density estimator (a technique to estimate the prob. distribution of a set of data points, which is non-parametric) to compute the likelihood of a set of parameters being good or bad. It first samples a few random combinations of parameters. Then it divides them into two groups: good and bad. Finally, TPE estimates the probability distributions of hyperparameter combinations for both good and bad groups using the Parzen estimator technique.
- CMA-ES: is used in the case in which we have continuous variables and when the search space is non-linear and non-convex. It is an example of evolutionary algorithm (EA), which aims to find the best solution to a problem by mimicking how nature evolves species through selection, reproduction, mutation, and inheritance. This method is well perform when we have a complex and non-linear search space or the evaluation of the validation is noisy such as when the metric is an inconsistent performance indicator.

  The starting point is a population of candidates. Then it modifies the candidates with each subsequent generation to adapt more closely to the best solution. CMA-ES applies the evolutionary principles as follows:
  - Within the hyperparameter search space, initialize the mean and the covariance matrix.
  - Repeat the evolutionary process:
    - Generate a population of candidates from the search space using the mean and the covariance matrix. Each candidate represents a combination of hyperparameter values.
    - Evaluate the fitness of the candidates. Fitness refers to the quality of a candidate or how well it solves the optimization problem. With CMA-ES, this means training the model on the dataset using the candidate hyperparameters and evaluating the performance on the validation set.
    - Select the best candidates from the population.
    - Update the mean and the covariance matrix from the best candidates.
    - Repeat for a maximum number of trials or until no improvement is seen in the population’s fitness.
   
_Comparison between TPE and CMA-ES_

The main differences between TPE and CMA-ES lie in their overall approach. TPE is a probabilistic model with a sequential search strategy, compared to CMA-ES, which is population-based and evaluates solutions in parallel. This often means TPE is more exploitative in its search, while CMA-ES balances exploration and exploitation using population control mechanisms. However, TPE is typically more efficient than CMA-ES, especially for a small number of parameters.
 
__Pruning__

Optuna also provides pruning strategy to avoid spending time on unpromising trails. Pruning occurs synchronously with the model training process: the validation error is checked during training, and the training is stopped if the algorithm is underperforming. In this way, pruning is similar to early stopping.
- Median pruning: each trial reports an intermediate result after n steps. The median of the intermediate results is then taken, and any trials below the median of previous trials at the same step are stopped.
- Successive halving: takes a more global approach and assigns a small, equal budget of training steps to all trials. Successive halving then proceeds iteratively: at each iteration, the performance of each trial is evaluated, and the top half of the candidates are selected for the next round, with the bottom half pruned away. The training budget is doubled for the next iteration, and the process is repeated. This way, the optimization budget is spent on the most promising candidates. As a result, a small optimization budget is spent on eliminating the underperforming candidates, and more resources are spent on finding the best parameters.
- Hyperband: extends successive halving by incorporating random search and a multi-bracket resource allocation strategy. It uses a multi-bracket resource allocation strategy, which divides the total computational budget into several brackets, each representing a different level of resource allocation. Within each bracket, successive halving is applied to iteratively eliminate underperforming configurations and allocate more resources to the remaining promising ones. At the beginning of each bracket, a new set of hyperparameter configurations is sampled using random search, which allows Hyperband to explore the hyperparameter space more broadly and reduce the risk of missing good configurations. This concurrent process enables Hyperband to adaptively balance exploration and exploitation in the search process, ultimately leading to more efficient and effective hyperparameter tuning.






__Implementing optuna for optimizing LightGBM hyperparameters__

- we need to define a set of objective/s for a study. Optuna passes a trial object to the objective function, which we can use to set up the parameters for the specific trial.
- We can save the study at different stages, and continue running the study from the saved point.
- A study could have single or multiple objectives. An example for single objective is to minimize f1-score. An example for multi-objective optimization is to minimize f1-score while having the highest learning rate (faster training).
- There are several visualization options to examine the result of the optimization:
  - Pareto front
  - Parallel coordinate plot
  - Parameter importance and plot it
 
__Define hyperparameter range__

Int he following example, all three types of hyperparameter are shown:
1. categorical: `trial.suggest_categorical('<name of hyperparameter>', '<list of parameters>')`
2. int: `trial.suggest_int('<name of hyperparameter>', low= ,high=, step=, log=)`
3. float: `trial.suggest_float('<name of hyperparameter>',  low=, high= , step=, log=)`

__Log scaling for numeric hyperparameters__

Numeric parameters (int and float) can have the option to log scale the range of possible values. Log scaling the parameter range has the effect that more values are tested close to the range’s lower bound and (logarithmically) fewer values towards the upper bound. Log scaling is particularly well suited to the learning rate where we want to focus on smaller values and exponentially increase tested values until the upper bound.

__Example__
```python
def objective(trial):
  boosting_type = trial.suggest_categorical("boosting_type", ["dart", "gbdt"])
  lambda_l1= trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
  min_child_samples= trial.suggest_int('min_child_samples', 5, 100),
  learning_rate = trial.suggest_float("learning_rate", 0.0001, 0.5, log=True),
  ...
```

__Apply pruning__

To apply pruning, we need to define callback and integrate it with the optimization. We need to specify the error metric (for example `binary`) in the call back:
`pruning_callback = optuna.integration.LightGBMPruningCallback(trial, "binary")`


__Define the model__

We define the model in a standard fashion, and pass callback and hyperparameters to it. 

Example:
```python
model = lgb.LGBMClassifier(
    force_row_wise=True,
    boosting_type=boosting_type,
    n_estimators=n_estimators,
    lambda_l1=lambda_l1,
    lambda_l2=lambda_l2,
    num_leaves=num_leaves,
    feature_fraction=feature_fraction,
    bagging_fraction=bagging_fraction,
    bagging_freq=bagging_freq,
    min_child_samples=min_child_samples,
    learning_rate=learning_rate,
    max_bin=max_bin,
    callbacks=[pruning_callback],
    verbose=-1)
```

__Trian the model__

Training the model follwos the standard process.

__Employ ptimization__

To run the optimization with objective function that was defined previously, we need to initialize sampler, pruner, and study. Then call optimize with our objective function: 
```python
sampler = optuna.samplers.TPESampler()
pruner = optuna.pruners.HyperbandPruner(
    min_resource=10, max_resource=400, reduction_factor=3)
study = optuna.create_study(
    direction='maximize', sampler=sampler,
    pruner=pruner
)
study.optimize(objective(), n_trials=100, gc_after_trial=True, n_jobs=-1)
```

In the above code, TPE is employed as sampler. The minimum and maximum resources specified for the Hyperband pruner control the minimum and the maximum number of iterations (or estimators) trained per trial. When applying pruning, the reduction factor controls how many trials are promoted in each halving round. The study is created by specifying the optimization direction (maximize or minimize).  In the above code, becasue the objective is _F1-Score_, the direction is maximize. `gc_after_trial` performs `gc.collect()` after each trial.

To get the best trial and parameters we can run this code: `print(study.best_trial)`



### saving and resuming an optimization study
Optuna provides two methods for saving and resuming an optimization study: 1.in memory and 2. using a remote database (RDB). When a study is run in memory, the standard Python methods for serializing an object such as joblib or pickle can be used:
- ```python
    joblib.dump(study, "lgbm-optuna-study.pkl")
  ```

- ```python
  study = joblib.load("lgbm-optuna-study.pkl")
  study.optimize(objective(), n_trials=20, gc_after_trial=True, n_jobs=-1)
  ```

RDB is an alternative approach for saving optimization study, in which the study’s intermediate (trial) and final results are persisted in a SQL database backend. The RDB can be hosted on a separate machine. Any of the SQL databases supported by SQL Alchemy may be used. 

__Example: saving a study using SQLite__
```python
study_name = "lgbm-tpe-rdb-study"
storage_name = f"sqlite:///{study_name}.db"
study = optuna.create_study(
    study_name=study_name,
    storage=storage_name,
    load_if_exists=False,
    sampler=sampler,
    pruner=pruner)
```

__Example: loading a study__
```python
study = optuna.create_study(study_name=study_name, storage=storage_name, load_if_exists=True)
```

### Visualization aid in Otuna
Optuna provides several visualization aids, among which is hyperparameter importance. It is a helpful tool for investigating hyperparameters and perhaps eliminating less important ones, or put more emphasize and resources for optimizing the more important ones. 
```python
fig = optuna.visualization.plot_param_importances(study)
fig.show()
```

Also we can employ parallel coordinate plot to visualize the interaction between parameters that are more important:
```python
fig = optuna.visualization.plot_parallel_coordinate(study, params=["boosting_type", "feature_fraction", "learning_rate", "n_estimators"])
```

### Multi-objective optimization
Optuna has the option for performing multi-objective optimization. We can defne several objectives, such as _F1-score_ (largest) and _number of leaves_ (smallest), with Optuna. We first need to define th emulti-objective function, and then perform the training. 
```python
def moo_objective(trial):
    learning_rate = trial.suggest_float("learning_rate", 0.0001, 0.5, log=True),
    model = lgb.LGBMClassifier(
        force_row_wise=True,
        boosting_type='gbdt',
        n_estimators=200,
        num_leaves=6,
        bagging_freq=7,
        learning_rate=learning_rate,
        max_bin=320,
    )
    scores = cross_val_score(model, X, y, scoring="f1_macro")
    return learning_rate[0], scores.mean()

# set the firection for both objectives during th etrianing phase:
study = optuna.create_study(directions=["maximize", "maximize"])
study.optimize(moo_objective, n_trials=100)
```

__Visualize the resutls__

To visualize the trade-off between two objectives, we can employ Pareto Front plot. 










  
---
# GPU-based and distributed learning with LightGBM

## Distributed learning with LightGBM and Dask

__Dask__ in a library for distributed computing and integrates seamlessly with Python libraries, inclugin sklearn and lightGBM [Ref.](https://www.dask.org/). It allows to set up clusters on a single machine, or across several machines. 

To run a local cluster we need to create local cluster object and assign alient to it:
```python
cluster = LocalCluster(n_workers=4, threads_per_worker=2) # cluster with 4 workers and 2 threads per worker
client = Client(cluster)
```
The cluster runs on localhost, with the scheduler running on port 8786 by default. Dask has a diagnostic dashboard, implemented with Bokeh.

Dask has its own dataframe implementation, called Dask DataFrane, compromises many smaller pandas Dataframes. 
```python
import dask.dataframe as dd
df = dd.read_csv("data.csv", blocksize="64MB")
print(f" Rows in dataframe: {df.shape[0].compute}")

# split dataset into X,y
X = df.iloc[:, :-1]
y = df.iloc[:, -1]
X_train, X_test, y_train, y_test = dask_ml.model_selection.train_test_split(X, y)

# employ LightGBM to classify
dask_model = lgb.DaskLGBMClassifier(n_estimators=200, client=client)
dask_model.fit(X_train, y_train)

# Make prediction
predictions = dask_model.predict(X_test)
#display results
predictions.compute()

```
__Note__ if the memory is large enough to fit a panda dataframe, it runs faster than Dast dataframe. 

Dask has three LightGBm learner implementations: `DaskLGBMRegressor`, `DaskLGBMClassifier`, and `DaskLGBMRanker`.
Dask LightGBM models can be fully serialized using Pickle or joblib.

LightGBM uses a Reduce-Scatter strategy for paralell computation:
- During the histogram-building phase, each worker builds histograms for different non-overlapping features. Then, a Reduce-Scatter operation is performed: each worker shares a part of its histogram with each other worker.
- After the Reduce-Scatter, each worker has a complete histogram for a subset of features and then finds the best split for these features.
- Finally, a gathering operation is performed: each worker shares its best split with all other workers, so all workers have all the best splits.
- The best feature split is chosen, and the data is partitioned accordingly.

## Train LightGBM on GPU

To train LightGBM, we can use two platforms: OpenCL and CUDA. To train LightGBM on GPU, we need to specify the device type in parameters. 

```python
model = lgb.LGBMClassifier(
        n_estimators=150,
        device="cuda",
        is_enable_sparse=False # it is not supported on GPU devices
)
model = model.fit(X_train, y_train)
```

__Best practive training with GPU__
- Always verify that the GPU is being used. LightGBM returns to CPU training if the GPU is unavailable despite setting device=gpu. A good way of checking is with a tool such as nvidia-smi, as shown previously, or comparing training times to reference benchmarks.
- Use a much smaller max_bin size. Large datasets reduce the impact of a smaller max_bin size, and the smaller number of bins benefits training on the GPU. Similarly, use single-precision floats for increased performance if your GPU supports it.
- GPU training works best for large, dense datasets. Data needs to be moved to the GPU’s VRAM for training, and if the dataset is too small, the overhead involved with moving the data is too significant.
- Avoid one-hot encoding of feature columns, as this leads to sparse feature matrices, which do not work well on the GPU.
- If employ automated optimization framework such as Optuna, set `n_jobs` to 1, when performing study optimization. Running running parallel jobs leveraging the GPU could cause unnecessary contention and overhead. `study.optimize(objective(), n_trials=10, gc_after_trial=True, n_jobs=1)`

-  



