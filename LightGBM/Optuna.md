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
- __Median pruning__: each trial reports an intermediate result after $n$ steps. The median of the intermediate results is then taken, and any trials below the median of previous trials at the same step are stopped.
- __Successive halving__: takes a more global approach and assigns a small, equal budget of training steps to all trials. Successive halving then proceeds iteratively: at each iteration, the performance of each trial is evaluated, and the top half of the candidates are selected for the next round, with the bottom half pruned away. The training budget is doubled for the next iteration, and the process is repeated.

  Thos method relies on a fixed initial set of configurations and a single resource allocation scheme.
  
- __Hyperband:__  extends successive halving by incorporating random search and a multi-bracket resource allocation strategy. It  uses a multi-bracket resource allocation strategy, which divides the total computational budget into several brackets, each representing a different level of resource allocation. Within each bracket, successive halving is applied to iteratively eliminate underperforming configurations and allocate more resources to the remaining promising ones. At the beginning of each bracket, a new set of hyperparameter configurations is sampled using random search, which allows Hyperband to explore the hyperparameter space more broadly and reduce the risk of missing good configurations. This concurrent process enables Hyperband to adaptively balance exploration and exploitation in the search process, ultimately leading to more efficient and effective hyperparameter tuning. According to [benchmark study by optuna](https://github.com/optuna/optuna/wiki/Benchmarks-with-Kurobako), hyperband is the among pruning method, employ witehr with TPE or CMA-ES.

# Optimization with optuna

Process of optimization with optuna starts with __1. defining an objective__. The objective function is called once for each trial. Optuna passes a trial object to the objective function, which we can use to set up the parameters for the specific trial. An exmaple of objective function can be maxmizing the f1-score. In the objective function, we can define potential values for each hp, (int, float, boolean, or categories) as follow:

```python
def objective(trial):
  boosting_type = trial.suggest_categorical("boosting_type", ["dart", "gbdt"])
  lambda_l1= trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True), # log-scale the range: more values are tested close to the range’s lower bound
  ...
  min_child_samples= trial.suggest_int('min_child_samples', 5, 100),
  learning_rate = trial.suggest_float("learning_rate", 0.0001, 0.5, log=True),
  max_bin = trial.suggest_int("max_bin", low = 128, high = 512, step = 32)
  n_estimators = trial.suggest_int("n_estimators", low = 40, high = 400, step = 20)
```

__2. defining pruning__ in optuna is perform by defining `pruning_callback`, which is integrated with the optimization process. This requires to _define error metric_. Example: `pruning_callback =  optuna.integration.LightGBMPruningCallback(trial, "binary")`.

__3. Fitting the model__ by passing parameters and call back as done normally. 
```ptyhpn
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
# train the model using five-fold cross validation w. F1 macro score
scores = cross_val_score(model, X, y, scoring="f1_macro")
# objective function to return mean of the F1-scores.
return scores.mean()
```

__3. Final step is to define samplers (TPE or CMA-ES), pruner, study, and call optimization__. For pruning, in case of choosing hyperband method, we need to define upper and lower band of resources, which controls number of iterations. The reduction factor in pruning controls how many trials are promoted in each halving round. When creating a study, we need to define direction (maximize or minimize) according to the objective. 

__4. to get results__ we need to call `study.best_trial`.

### Saving and resuming optimization
Optimizing hps may take a long time. Saving optimization study and resuming it later is done in two ways: 
1. in memory, with standard python methods for serializung object, such as `joblib` or `pickle`. Example: `joblib.dump(study, "lgbm-optuna-study.pkl")`

  To restore and continue the optimization, we can then load the saved study and continue the optimization. Example:
  ```python
  study = joblib.load("lgbm-optuna-study.pkl")
  study.optimize(objective(), n_trials=20, gc_after_trial=True, n_jobs=-1)
  ```
2. using remote database: When using an RDB, the study’s intermediate (trial) and final results are persisted in a SQL database backend. The RDB can be hosted on a separate machine. Any of the SQL databases supported by SQL Alchemy may be used.

  Example:
  ```python
  # SQLite as an RDB
  study_name = "lgbm-tpe-rdb-study"
  storage_name = f"sqlite:///{study_name}.db"
  study = optuna.create_study(
                              study_name=study_name,
                              storage=storage_name,
                              load_if_exists=False,
                              sampler=sampler,
                              pruner=pruner)
```
To restore the optimization, we need to specify the same storage ad set `load_if_exist = True`:

```python
study = optuna.create_study(study_name=study_name,
storage=storage_name, load_if_exists=True)
```

### Analyze results: parameter effects
After the initial round of optimization, analyszing the results helps to identify those paramters that have more effect on the model performance. For the next round of optimization, we can then perform optimization of those parameters with smaller granuality. 

A strightforward visualization technique is employing built-in visualization feature of optuna. Example:
```python
# visuaalize the parameter importance as barchart
fig = optuna.visualization.plot_param_importances(study)
fig.show()
```
Another approach is employing parallel coordinate plot. This approach requires to specify which parameters we would like the plot to contain. Example:

```python
fig = optuna.visualization.plot_parallel_coordinate(study, params=
["boosting_type", "feature_fraction", "learning_rate", "n_estimators"])
fig.show()
```

## Multi-object optimization


 
