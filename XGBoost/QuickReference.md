# XGBoost quick reference

## XGBoost regressor parameters - layer
The folowing is the list of XGBoost parameters in layers divided by their impact on the model performance. I use this list in optuna for hyper-parameters tuning step by step, from high level to more granual level for fine-tuning.

```python
 es = callback.EarlyStopping(rounds=50,
                                        #min_delta=1e-3,
                                        #save_best=True,
                                        #maximize=False,
                                        #data_name="validation_0",
                                        #metric_name= 'rmse',
                                        )
    params = {
        # First layer -  initial parameters to tune
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'max_depth': trial.suggest_int('max_depth', 4, 12),
        'n_estimators': trial.suggest_int('n_estimators', 500, 2000),
        
        ## Second layer parameters to tune: reduce overfitting
        #'subsample': trial.suggest_float('subsample', 0.5, 1.0, log=True),
        #'colsample_bytree': trial.suggest_float('colsample_bytree', 0.1, 1.0),
        
        ## Third layer - fine-tune - regulaization parameters
        #'reg_lambda': trial.suggest_float('reg_lambda', 0.01, 10.0, log=True),
        #'reg_alpha': trial.suggest_float('reg_alpha', 0.01, 10.0, log=True),
        #'gamma': trial.suggest_float('gamma', 0.01, 10.0, log=True),
        #'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        
        'objective': 'reg:squarederror',
        'device': 'cuda',           # Use GPU for training
        'tree_method': 'gpu_hist',  # Use GPU for training
        'random_state': 42,
        'enable_categorical': True,  # Enable categorical feature support
        'callbacks': [es]
    }
```



---
### XGBoost parameters complete
```
n_estimators : Optional[int]
    Number of gradient boosted trees.  Equivalent to number of boosting
    rounds.

max_depth :  Optional[int]
    Maximum tree depth for base learners.
max_leaves :
    Maximum number of leaves; 0 indicates no limit.
max_bin :
    If using histogram-based algorithm, maximum number of bins per feature
grow_policy :
    Tree growing policy. 0: favor splitting at nodes closest to the node, i.e. grow
    depth-wise. 1: favor splitting at nodes with highest loss change.
learning_rate : Optional[float]
    Boosting learning rate (xgb's "eta")
verbosity : Optional[int]
    The degree of verbosity. Valid values are 0 (silent) - 3 (debug).
objective : typing.Union[str, typing.Callable[[numpy.ndarray, numpy.ndarray], typing.Tuple[numpy.ndarray, numpy.ndarray]], NoneType]
    Specify the learning task and the corresponding learning objective or
    a custom objective function to be used (see note below).
booster: Optional[str]
    Specify which booster to use: gbtree, gblinear or dart.
tree_method: Optional[str]
    Specify which tree method to use.  Default to auto.  If this parameter is set to
    default, XGBoost will choose the most conservative option available.  It's
    recommended to study this option from the parameters document :doc:`tree method
    </treemethod>`
n_jobs : Optional[int]
    Number of parallel threads used to run xgboost.  When used with other
    Scikit-Learn algorithms like grid search, you may choose which algorithm to
    parallelize and balance the threads.  Creating thread contention will
    significantly slow down both algorithms.
gamma : Optional[float]
    (min_split_loss) Minimum loss reduction required to make a further partition on a
    leaf node of the tree.
min_child_weight : Optional[float]
    Minimum sum of instance weight(hessian) needed in a child.
max_delta_step : Optional[float]
    Maximum delta step we allow each tree's weight estimation to be.
subsample : Optional[float]
    Subsample ratio of the training instance.
sampling_method :
    Sampling method. Used only by the GPU version of ``hist`` tree method.
      - ``uniform``: select random training instances uniformly.
      - ``gradient_based`` select random training instances with higher probability
        when the gradient and hessian are larger. (cf. CatBoost)
colsample_bytree : Optional[float]
    Subsample ratio of columns when constructing each tree.
colsample_bylevel : Optional[float]
    Subsample ratio of columns for each level.
colsample_bynode : Optional[float]
    Subsample ratio of columns for each split.
reg_alpha : Optional[float]
    L1 regularization term on weights (xgb's alpha).
reg_lambda : Optional[float]
    L2 regularization term on weights (xgb's lambda).
scale_pos_weight : Optional[float]
    Balancing of positive and negative weights.
base_score : Optional[float]
    The initial prediction score of all instances, global bias.
random_state : Optional[Union[numpy.random.RandomState, int]]
    Random number seed
missing : float, default np.nan
    Value in the data which needs to be present as a missing value.
num_parallel_tree: Optional[int]
    Used for boosting random forest.
monotone_constraints : Optional[Union[Dict[str, int], str]]
    Constraint of variable monotonicity.  See :doc:`tutorial </tutorials/monotonic>`
    for more information.
interaction_constraints : Optional[Union[str, List[Tuple[str]]]]
    Constraints for interaction representing permitted interactions.  The
    constraints must be specified in the form of a nested list, e.g. ``[[0, 1], [2,
    3, 4]]``, where each inner list is a group of indices of features that are
    allowed to interact with each other.  See :doc:`tutorial
    </tutorials/feature_interaction_constraint>` for more information
importance_type: Optional[str]
    The feature importance type for the feature_importances\_ property:

    * For tree model, it's either "gain", "weight", "cover", "total_gain" or
      "total_cover".
    * For linear model, only "weight" is defined and it's the normalized coefficients
      without bias.

device : Optional[str]

    .. versionadded:: 2.0.0

    Device ordinal, available options are `cpu`, `cuda`, and `gpu`.

validate_parameters : Optional[bool]

    Give warnings for unknown parameter.

enable_categorical : bool

    .. versionadded:: 1.5.0

    .. note:: This parameter is experimental

    Experimental support for categorical data.  When enabled, cudf/pandas.DataFrame
    should be used to specify categorical data type.  Also, JSON/UBJSON
    serialization format is required.

feature_types : Optional[FeatureTypes]

    .. versionadded:: 1.7.0

    Used for specifying feature types without constructing a dataframe. See
    :py:class:`DMatrix` for details.

max_cat_to_onehot : Optional[int]

    .. versionadded:: 1.6.0

    .. note:: This parameter is experimental

    A threshold for deciding whether XGBoost should use one-hot encoding based split
    for categorical data.  When number of categories is lesser than the threshold
    then one-hot encoding is chosen, otherwise the categories will be partitioned
    into children nodes. Also, `enable_categorical` needs to be set to have
    categorical feature support. See :doc:`Categorical Data
    </tutorials/categorical>` and :ref:`cat-param` for details.

max_cat_threshold : Optional[int]

    .. versionadded:: 1.7.0

    .. note:: This parameter is experimental

    Maximum number of categories considered for each split. Used only by
    partition-based splits for preventing over-fitting. Also, `enable_categorical`
    needs to be set to have categorical feature support. See :doc:`Categorical Data
    </tutorials/categorical>` and :ref:`cat-param` for details.

multi_strategy : Optional[str]

    .. versionadded:: 2.0.0

    .. note:: This parameter is working-in-progress.

    The strategy used for training multi-target models, including multi-target
    regression and multi-class classification. See :doc:`/tutorials/multioutput` for
    more information.

    - ``one_output_per_tree``: One model for each target.
    - ``multi_output_tree``:  Use multi-target trees.

eval_metric : Optional[Union[str, List[str], Callable]]

    .. versionadded:: 1.6.0

    Metric used for monitoring the training result and early stopping.  It can be a
    string or list of strings as names of predefined metric in XGBoost (See
    doc/parameter.rst), one of the metrics in :py:mod:`sklearn.metrics`, or any other
    user defined metric that looks like `sklearn.metrics`.

    If custom objective is also provided, then custom metric should implement the
    corresponding reverse link function.

    Unlike the `scoring` parameter commonly used in scikit-learn, when a callable
    object is provided, it's assumed to be a cost function and by default XGBoost will
    minimize the result during early stopping.

    For advanced usage on Early stopping like directly choosing to maximize instead of
    minimize, see :py:obj:`xgboost.callback.EarlyStopping`.

    See :doc:`Custom Objective and Evaluation Metric </tutorials/custom_metric_obj>`
    for more.

    .. note::

         This parameter replaces `eval_metric` in :py:meth:`fit` method.  The old
         one receives un-transformed prediction regardless of whether custom
         objective is being used.

    .. code-block:: python

        from sklearn.datasets import load_diabetes
        from sklearn.metrics import mean_absolute_error
        X, y = load_diabetes(return_X_y=True)
        reg = xgb.XGBRegressor(
            tree_method="hist",
            eval_metric=mean_absolute_error,
        )
        reg.fit(X, y, eval_set=[(X, y)])

early_stopping_rounds : Optional[int]

    .. versionadded:: 1.6.0

    - Activates early stopping. Validation metric needs to improve at least once in
      every **early_stopping_rounds** round(s) to continue training.  Requires at
      least one item in **eval_set** in :py:meth:`fit`.

    - If early stopping occurs, the model will have two additional attributes:
      :py:attr:`best_score` and :py:attr:`best_iteration`. These are used by the
      :py:meth:`predict` and :py:meth:`apply` methods to determine the optimal
      number of trees during inference. If users want to access the full model
      (including trees built after early stopping), they can specify the
      `iteration_range` in these inference methods. In addition, other utilities
      like model plotting can also use the entire model.

    - If you prefer to discard the trees after `best_iteration`, consider using the
      callback function :py:class:`xgboost.callback.EarlyStopping`.

    - If there's more than one item in **eval_set**, the last entry will be used for
      early stopping.  If there's more than one metric in **eval_metric**, the last
      metric will be used for early stopping.

    .. note::

        This parameter replaces `early_stopping_rounds` in :py:meth:`fit` method.

callbacks : Optional[List[TrainingCallback]]
    List of callback functions that are applied at end of each iteration.
    It is possible to use predefined callbacks by using
    :ref:`Callback API <callback_api>`.

    .. note::

       States in callback are not preserved during training, which means callback
       objects can not be reused for multiple training sessions without
       reinitialization or deepcopy.

    .. code-block:: python

        for params in parameters_grid:
            # be sure to (re)initialize the callbacks before each run
            callbacks = [xgb.callback.LearningRateScheduler(custom_rates)]
            reg = xgboost.XGBRegressor(**params, callbacks=callbacks)
            reg.fit(X, y)

kwargs : dict, optional
    Keyword arguments for XGBoost Booster object.  Full documentation of parameters
    can be found :doc:`here </parameter>`.
    Attempting to set a parameter via the constructor args and \*\*kwargs
    dict simultaneously will result in a TypeError.

    .. note:: \*\*kwargs unsupported by scikit-learn

        \*\*kwargs is unsupported by scikit-learn.  We do not guarantee
        that parameters passed via this argument will interact properly
        with scikit-learn.

    .. note::  Custom objective function

        A custom objective function can be provided for the ``objective``
        parameter. In this case, it should have the signature
        ``objective(y_true, y_pred) -> grad, hess``:

        y_true: array_like of shape [n_samples]
            The target values
        y_pred: array_like of shape [n_samples]
            The predicted values

        grad: array_like of shape [n_samples]
            The value of the gradient for each sample point.
        hess: array_like of shape [n_samples]
            The value of the second derivative for each sample point
```
