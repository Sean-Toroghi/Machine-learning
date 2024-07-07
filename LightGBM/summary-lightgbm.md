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
- control min number of samples per leaf: 

 
---
## LightGBM

---
