<h1>Parameter optimization with Optuna</h1>

References
- [optuna](https://optuna.org/)
- [Machine Learning with LightGBM and Python - 2023](https://www.oreilly.com/library/view/machine-learning-with/9781800564749/)


# Hyper-parameter (hp) tuning
Choosing the best hps for a problem is crucial for ensemble tree-based method, which impacts both model performance and its capabilities to generalize. However, it is a challenging task, as it imposes high cost to test each combination of hps, while the size of search space is very large. What makes it even harder to perfrom is the interaction between hps, as can be observed by ploting a prallel coordinate plot. The process of finding optimal parameters is called _study_ and each configuration is called a _trial_. 

Some of the methods for finding optimal hps are:
- GridSearch: this method gets a range for each hp, and perfom an exhaustive search over the combination of all hps. The downside of this method is its cost.
- SHERPA
- Hyperopt
- Talos
- Optuna: Optuna provides efficient optimization algorithms to search hyperparameter spaces more effectively. In addition to the optimization algorithms, Optuna also provides pruning strategies to save computational resources and time by pruning poorly performing trials.

# Optuna
Optuna has several optimization alogorithms for searching hp space and find the optimal combination. Among these algorithms are Tree-Structured Parzen Estimator
(TPE) and a Covariance Matrix Adaptation Evolution Strategy (CMA-ES). Furthermore, it has pruning strategy that prunes poorly perform trials, leading to more efficient use of resources. 

## Optimiztion algorithms in Optuna

### TPE
TPE starts by sampling a few random combinations of hyperparameters and evaluating the model’s performance for each. Based on these initial results, TPE divides the hyperparameter combinations into two groups: good (those that lead to better performance) and bad (those that lead to worse performance):
- l(x): The probability density function of good configurations
- g(x) : The probability density function of bad configurations

TPE then estimates the probability distributions of hyperparameter combinations for both good and bad groups using the Parzen estimator technique. With estimations of the probability distributions available, TPE calculates the Expected Improvement (EI) of hyperparameter configurations. EI can be calculated as the ratio between the two densities: $\frac{l(x)}{g(x)}$
