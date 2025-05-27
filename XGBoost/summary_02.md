x# XGBoost - summary (v02) and key take aways 

References:
- Book (2024): XGBoost for Regression Predictive Modeling and Time Series Analysis
- Paper (2016): XGBoost: A Scalable Tree Boosting System [https://arxiv.org/abs/1603.02754]
- Website: XGBoost [link](https://xgboost.readthedocs.io/en/stable/index.html)

This summary is my takeaways and notes from reference book "XGBoost for Regression Predictive Modeling and Time Series Analysis" by Partha Pritam Deka and Joyce Weiner that was published in 2024, the XGboost website [link](https://xgboost.readthedocs.io/en/stable/index.html), and the original published paper in 2016 XGBoost: A Scalable Tree Boosting System [arxiv](https://arxiv.org/pdf/1603.02754).


---
# Paper: "XGBoost: A Scalable Tree Boosting System" 2016 [arxiv](https://arxiv.org/pdf/1603.02754)

XGBoost was developed to address the following shorcommings of existing ensemble tree-boosting algorithms at the time in 2016: ability to handle large-scale data, flexibility in defining customized optimization objectives and evaluation criteria, and support for parallel processing and distributed computing. The XGBoost algorithm could address the shortcomming by adding some enhancements to the gradient-boosted trees:
- a sparsity-aware algorithm for handling missing values 
- a regularization term to control model complexity
- eficiency and scalability, achieved through a cache-aware block structure and parallel tree construction

The list of improvements applied to the CART are:
- Novel tree learning algorithm
- Sparse data handling
- New algorithm for finding proposed split points
- Enabled parallel and distributed compute
- Cache-aware algorithm to prevent memory-related delays 

## How boosting improves the performance of a classification and regression tree (CART) model 

In boosting, results from multiple iterations are combined to improve the overall classification or prediction. Boosting is an iterative method of combining multiple “weak learners” (a model that is just slightly better than guessing) into a “strong learner.”

## How gradient descent works in XGBoost
Gradient descent is an algorithm that minimizes a function in an iterative manner. It is used to calculate the weights for the X variables in the loss function, comparing the predicted value to the actual from the training dataset.

For classification decision trees, Gini impurity is used in the CART algorithm. It reaches zero when all items in a node fall into a single classification.

## Issue of sparse data and how XGBoost addressed it

To address sparse data issue (either caused by missing values, or as the result of encoding such as one-hot-encoding), XGBoost uses sparse-aware spliting technique. This technique provides a default split direction when it faces sparse data, which speeds up processing time.

## Memory issue, and how XGBoost run efficient

The greedy algorithm to find where to split the data requires all data to fit in memory. XGBoost employs approximation to handle issue of memory allocation when it comes to large dataset. This approximation method first propose split candidates are first based on feature distributions (what the histograms of the individual feature columns look like), then the features are mapped into buckets split by those candidates. Finaly the algorithm picks the best split based on aggregated statistics. XGBoost uses two approximation methods:
1. global: it uses the same split candidates from the initial mapping throughout the steps,
2. local: refine candidates after each split


## Issue of overfititng and how XGBoost addressed it
The XGBoost algorithm uses _omega_ function to smooth the weights and as the result, avoids overfitting. Omega function acts as a regularization and controls the complexity of the model. Furthermore, XGBoost employs two additional techniques to handle overfitting:
- __shrinkage__: scale weights after each step of tree boosting
- __subsampling__: perform subsampling on both rows and features (similar to random forest)


---

# tree based and ensemble models

## Gradient-boosted trees

Gradient-boosted trees are a type of classification and regression tree (CART) model, that learns via building a decision tree. The algorihtm employs the gradient descent algorithm to minimizing a loss function that compares the predicted value versus a target value.

## XGBoost vs Random forest

While in gradient-boosted trees such as XGBoost, the decision trees are built iteratively, random forest algorithm builds multiple decision trees at the same time. Random forest uses a random sampling of the dataset for each tree, and the results of the trees in the forest are aggregated to produce the final result. In contrast, gradient-boosted trees aggregate the results as it goes through the training process.

The following considerations could help in making decision which algorithm to pick:
- The iterative nature of gradient-boosted trees makes it harder to explain why the model is making a prediction since the results process is an aggregate of multiple iterations. On the other hand, random forest results is easier to interpret.
- Data structure: with wide data (many feature), random forest algorithm performs better than gradient-boosted trees. Gradient boosting works well for tall data, where there are a lot of rows in the dataset.

---

__Evaluation metrics__

XGboost provides a range of evaluation [metrics](https://xgboost.readthedocs.io/en/stable/parameter.html#learning-task-parameters). The default metrics are: rmse for regression, and logloss for classification. Based on objective, the model picks the appropriate metric, if not specified. For example for "rank:map" the default metric is _mean average precision_.

While there are over 20 metrics, here is the list of ones that are used more frequently:
- rmse: root mean square error
- rmsle: root mean square log error: Default metric of reg:squaredlogerror objective. This metric reduces errors generated by outliers in dataset. But because log function is employed, rmsle might output nan when prediction value is less than -1.
- mae: mean absolute error
- mape: mean absolute percentage error
- mphe: mean Pseudo Huber error. Default metric of reg:pseudohubererror objective.
- logloss: negative log-likelihood
- error: Binary classification error rate. It is calculated as #(wrong cases)/#(all cases). For the predictions, the evaluation will regard the instances with prediction value larger than 0.5 as positive instances, and the others as negative instances.
- error@t: a different than 0.5 binary classification threshold value could be specified by providing a numerical value through ‘t’.
- merror: Multiclass classification error rate. It is calculated as #(wrong cases)/#(all cases).
- mlogloss: Multiclass logloss.
- auc: Receiver Operating Characteristic Area under the Curve. Available for classification and learning-to-rank tasks.
  - When used with binary classification, the objective should be binary:logistic or similar functions that work on probability.
  - When used with multi-class classification, objective should be multi:softprob instead of multi:softmax, as the latter doesn’t output probability. Also the AUC is calculated by 1-vs-rest with reference class weighted by class prevalence.
  - When used with LTR task, the AUC is computed by comparing pairs of documents to count correctly sorted pairs. This corresponds to pairwise learning to rank. The implementation has some issues with average AUC around groups and distributed workers not being well-defined.
  - On a single machine the AUC calculation is exact. In a distributed environment the AUC is a weighted average over the AUC of training rows on each node. therefore, distributed AUC is an approximation sensitive to the distribution of data across workers. Use another metric in distributed environments if precision and reproducibility are important.
  - When input dataset contains only negative or positive samples, the output is NaN. The behavior is implementation defined, for instance, scikit-learn returns instead.
- aucpr: Area under the PR curve. Available for classification and learning-to-rank tasks.
- pre: Precision at _k_. Supports only learning to rank task.
- ndcg: Normalized Discounted Cumulative Gain
- map: Mean Average Precision
- ndcg@n, map@n, pre@n: can be assigned as an integer to cut off the top positions in the lists for evaluation.
- ndcg-, map-, ndcg@n-, map@n-: In XGBoost, the NDCG and MAP evaluate the score of a list without any positive samples as 1. By appending “-” to the evaluation metric name, we can ask XGBoost to evaluate these scores as  to be consistent under some conditions.
- poisson-nloglik: negative log-likelihood for Poisson regression
- gamma-nloglik: negative log-likelihood for gamma regression
- cox-nloglik: negative partial log-likelihood for Cox proportional hazards regression
- gamma-deviance: residual deviance for gamma regression
- tweedie-nloglik: negative log-likelihood for Tweedie regression (at a specified value of the tweedie_variance_power parameter)
- aft-nloglik: Negative log likelihood of Accelerated Failure Time model. See Survival Analysis with Accelerated Failure Time for details.
- interval-regression-accuracy: Fraction of data points whose predicted labels fall in the interval-censored labels. Only applicable for interval-censored data (Survival Analysis with Accelerated Failure Time for details.)

---
