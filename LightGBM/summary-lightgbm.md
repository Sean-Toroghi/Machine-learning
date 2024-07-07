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
## LightGBM

---
