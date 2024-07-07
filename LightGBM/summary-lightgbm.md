# LightGBM

References
- [Machine Learning with LightGBM and Python by Andrich van Wyk](https://learning.oreilly.com/library/view/machine-learning-with/9781800564749/B16690_01.xhtml)

---
## Ovreview

Data preprocessign pipeline for a machine learning task usually follows these steps:


__Overfitting__ is a phenomenon in the ML field, occurs when a model memorize the data and fit the noise, leading to loose the generalization ability. Overfitting stem from one or combination of the following factors:
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
- F1-score is the harmonic mean between precision and recal.

__2- regression metrics__
- mean square error: average of the squared differences between predicted and actual values. While it is differenciable (can be used in gradient based learning), it penalizes large errors more heavilty that small errors (due to the squaring the difference).
- mean absolute error 
- average of the absolute differences between predicted and actual values. It is more robust against the size of errors and less sensitive to outliers. Unforetunately it is not differentiable!

  
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
- half Poisson deciance


__Advantages of tree-based models__
- they could use both numerical and categorical features
- less sensitive to data range and size, leads to reducing data preparation effort
- the result is interpretable

__Disadvantages of tree-based models__
- prone to overfitting
- perform poor at extrapolation tasks
- perform poor when trained on unbalanced data. The high-frequency classes will dominate the prediciton.

__Mitigate overfitting issue__: there are several strategies implmentedin tree-based models to overcome overfitting issue, among which are:
- pruning: removing vranches that do not contribute much information gain, leading to reduce model complexity.
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

To bring diversity to the ensemble, we can train models on subset of samples, subset of features, differnt models, differnt set of hyper-parameters, or differentiate by diversify a part of model (such as embedding, or ramdom initialization of parameters). Some of the most significant ensemble approaches are:
- bagging: train on subset of samples and features
- boosting: interatively train models on the error or the previous models
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

GBDT (also called multipleadditive regression trees (MART)) method sequentially train models, each learns from the mistakes of the previous models. Each model uses the entire dataset. Some of the hyper-parameters for graient bossting method are as follow ([ref](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html)):
- n_estimators: Controls the number of trees in the ensemble. Generally, more trees are better. However, a point of diminishing returns is often reached, and overfitting occurs when there are too many trees.
- learning_rate: Controls the contribution of each tree to the ensemble. Lower learning rates lead to longer training times and may require more trees to be built (larger values for n_estimators). Setting learning_rate to a very large value may cause the optimization to miss optimum points and must be combined with fewer trees. Then, the DART algorithm is to apply additional scaling of the contribution of the new tree.

__DART__

DART is an extension of GBDT, which employs dropouts to avoid overfitting. First, the next decision tree is built from a random subset of the previous tree, with probability $p$ that indicates probability of previous tree being included. DART is implemented as a part of LightGBM.

---
---

# LightGBM

---

## Overview

LightGBM is a gradient-boosting framework for tree-based ensemble method, with focus on efficiency (swpace and time), while improving accuracy. Applications with high-dim and large data size is where LightGBM shines. It supports both regression and classification (binary or multiclass) tasks, and ranking via LambdaRank. LightGBM employs several techniques and provides customization through hyper-parameter tuning, including DART, bagging, continous training, early stopping, and several other options.

__Optimization__

To improve model efficiency, Lightbgm benefits from __histogram sorting__ method:
- Reduce computation complexity via _histogram sorting_: the most computational intense task of a GBDT is training the regression tree for each iteration, where finding the optimal split is very expensive. Algorithm requires to sort the samples (either at prior to spliting, or at the time of spliting). By creating feature histograms, the time-complexity for building a decision node reduce from $O(n)$ (pre-sorting approach) to $O(b)$, where $b$ is number of bins.
- Employ _histogram subtraction_ for building the historams for the leaves. This approach subtracts the leaf’s neighbor’s histogram from the parent’s histogramn, instead of calculating the histogram for each leaf.
- Employ histogram to reduce space complexity (memory cost). Any sorting method requires memory allocation, while histogram sorting does mot. Also, a more efficient data type is required to store bins (number of bins is much smaller than the size of dataset).

LightGBM employs __exlusive feature bundling (EFB)__ to optimize working with sparse data.

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

While LightGBM has many hyper-parameters, they can be divided into core framework, accuracy related, and learning constrol (avoid overfitting) parameters:

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

### Implementation

__Validate model performance__

- __Cross validation__: an alternative to split data into train/val/test set, is cross-validation, in which the dataset splitting multiple times and train the model multiple times, once for each split.
- __Stratified k-fold validation__: preservs the percentage of samples for each class when creating folds. This way, all folds will have the same ditribution of classes as the original dataset.

__Parameter optimization__


















