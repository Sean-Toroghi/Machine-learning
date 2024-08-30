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

## Optimiztion and prunning algorithms in Optuna

### TPE
TPE starts by sampling a few random combinations of hyperparameters and evaluating the model’s performance for each. Based on these initial results, TPE divides the hyperparameter combinations into two groups: good (those that lead to better performance) and bad (those that lead to worse performance):
- l(x): The probability density function of good configurations
- g(x) : The probability density function of bad configurations

TPE then estimates the probability distributions of hyperparameter combinations for both good and bad groups using the Parzen estimator technique. With estimations of the probability distributions available, TPE calculates the Expected Improvement (EI) of hyperparameter configurations. EI can be calculated as the ratio between the two densities: $\frac{l(x)}{g(x)}$. With each trail, the algorithm samples new hyperparameter configurations that maximize the EI. TPE estimates the distributions of good and bad parameters and uses them to find optimal parameters by maximizing new trials’ expected improvement. TPE is cost-effective since it approximates the distributions and can search for better parameters optimally (in a non-exhaustive way). TPE also handles parameter interactions.

### CMA-ES
CMA-ES is another optimization algorithm in Optuna, which __is suitable for cases that involve continuous variables and when the search space is non-linear and nonconvex.__ This method is based on the evolutionary algorithm, which is an optimization method aiming to find the best solution to a problem by mimicking how nature evolves species through selection, reproduction, mutation, and inheritance. Evolutionary algorithms start with a population of candidate solutions and modify the candidates with each subsequent generation to adapt more closely to the best solution. 

Central to the evolutionary process of CMA-ES is the covariance matrix. A covariance matrix is a square, symmetric matrix representing the covariance between pairs of variables (the hyperparameters), providing insight into their relationships. The diagonal elements of the matrix represent the variances of individual variables, while the off-diagonal elements represent the covariances between pairs of variables.

CMA-ES applies the evolutionary principles as follows when optimizing hyperparameters:
1. Within the hyperparameter search space, initialize the mean and the covariance matrix.
2. Repeat the evolutionary process:
  1. Generate a population of candidates from the search space using the mean and the covariance matrix. Each candidate represents a combination of hyperparameter values.
  2. Evaluate the fitness of the candidates. Fitness refers to the quality of a candidate or how well it solves the optimization problem. With CMA-ES, this means training the model on the dataset using the candidate hyperparameters and evaluating the performance on the
validation set.
  3. Select the best candidates from the population.
  4. Update the mean and the covariance matrix from the best candidates.
  5. Repeat for a maximum number of trials or until no improvement is seen in the population’s fitness.

### TPE vs CMA-ES

TPE
- TPE is a probabilistic model with a sequential search strategy
- TPE is more exploitative in its search.
- TPE is typically more efficient than CMA-ES, especially for a small number of
parameters.

CMA-ES
- CMA-ES is population-based and evaluates solutions in parallel.
- CMA-ES balances exploration and exploitation using population control mechanisms.
- CMA-ES is suitable for cases that involve continuous variables and when the search space is non-linear and nonconvex.

### Pruning strategies
Pruning occurs
synchronously with the model training process: the validation error is checked during training, and the training is stopped if the algorithm is underperforming. In this way, pruning is similar to early stopping. Some of the pruning methods in optuna are:
- Median pruning: each trial reports an intermediate result after $n$ steps. The median of the intermediate results is then taken, and any trials below the median of previous trials at the same step are stopped.
- Successive halving: takes a more global approach and assigns a small, equal budget of training steps to all trials. Successive halving then proceeds iteratively: at each iteration, the performance of each trial is evaluated, and the top half of the candidates are selected for the next round, with the bottom half pruned away. The training budget is doubled for the next iteration, and the process is repeated.

  Thos method relies on a fixed initial set of configurations and a single resource allocation scheme.
  
- Hyperband:  extends successive halving by incorporating random search and a multi-bracket resource allocation strategy


 
