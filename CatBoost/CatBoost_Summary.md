<h1>CatBoost model summary</h1>

# Parameters
Perhaps, one important aspect, and at the same time a challenging part of modeling with the tree-based models such as Catboost is hyper-parameter tuning. Furthermore the number of hyper-parameters both gives them stregth and is a weakness. It enhances them to be customized for the task in hand, while at the same time, picking the right set of parameters is a challenging task. 

Here is a complete list of hyper-parameters for Catbosot classifier (Sklearn API), and a short summary for each one:

```python
class CatBoostClassifier(iterations=None,
                         learning_rate=None,
                         depth=None,
                         l2_leaf_reg=None,
                         model_size_reg=None,
                         rsm=None,
                         loss_function=None,
                         border_count=None,
                         feature_border_type=None,
                         per_float_feature_quantization=None,
                         input_borders=None,
                         output_borders=None,
                         fold_permutation_block=None,
                         od_pval=None,
                         od_wait=None,
                         od_type=None,
                         nan_mode=None,
                         counter_calc_method=None,
                         leaf_estimation_iterations=None,
                         leaf_estimation_method=None,
                         thread_count=None,
                         random_seed=None,
                         use_best_model=None,
                         verbose=None,
                         logging_level=None,
                         metric_period=None,
                         ctr_leaf_count_limit=None,
                         store_all_simple_ctr=None,
                         max_ctr_complexity=None,
                         has_time=None,
                         allow_const_label=None,
                         classes_count=None,
                         class_weights=None,
                         auto_class_weights=None,
                         one_hot_max_size=None,
                         random_strength=None,
                         name=None,
                         ignored_features=None,
                         train_dir=None,
                         custom_loss=None,
                         custom_metric=None,
                         eval_metric=None,
                         bagging_temperature=None,
                         save_snapshot=None,
                         snapshot_file=None,
                         snapshot_interval=None,
                         fold_len_multiplier=None,
                         used_ram_limit=None,
                         gpu_ram_part=None,
                         allow_writing_files=None,
                         final_ctr_computation_mode=None,
                         approx_on_full_history=None,
                         boosting_type=None,
                         simple_ctr=None,
                         combinations_ctr=None,
                         per_feature_ctr=None,
                         task_type=None,
                         device_config=None,
                         devices=None,
                         bootstrap_type=None,
                         subsample=None,
                         sampling_unit=None,
                         dev_score_calc_obj_block_size=None,
                         max_depth=None,
                         n_estimators=None,
                         num_boost_round=None,
                         num_trees=None,
                         colsample_bylevel=None,
                         random_state=None,
                         reg_lambda=None,
                         objective=None,
                         eta=None,
                         max_bin=None,
                         scale_pos_weight=None,
                         gpu_cat_features_storage=None,
                         data_partition=None
                         metadata=None,
                         early_stopping_rounds=None,
                         cat_features=None,
                         grow_policy=None,
                         min_data_in_leaf=None,
                         min_child_samples=None,
                         max_leaves=None,
                         num_leaves=None,
                         score_function=None,
                         leaf_estimation_backtracking=None,
                         ctr_history_unit=None,
                         monotone_constraints=None,
                         feature_weights=None,
                         penalties_coefficient=None,
                         first_feature_use_penalties=None,
                         model_shrink_rate=None,
                         model_shrink_mode=None,
                         langevin=None,
                         diffusion_temperature=None,
                         posterior_sampling=None,
                         boost_from_average=None,
                         text_features=None,
                         tokenizers=None,
                         dictionaries=None,
                         feature_calcers=None,
                         text_processing=None,
                         fixed_binary_splits=None)
```
Certainly! Here are the parameters of `CatBoostClassifier` explained one by one:

1. **iterations**: Number of boosting iterations. More iterations usually mean better accuracy but can lead to overfitting.
2. **learning_rate**: The rate at which the model learns. A smaller value makes learning slower but can result in better accuracy.
3. **depth**: Depth of the tree. Larger values can lead to overfitting.
4. **l2_leaf_reg**: L2 regularization term on weights. This helps to prevent overfitting by adding a penalty for larger weights.
5. **model_size_reg**: Regularization for the model size. This helps in controlling the size of the model.
6. **rsm**: Random subspace method. The fraction of features to be used for fitting each tree.
7. **loss_function**: Loss function used to evaluate model performance. Examples include 'Logloss' for classification and 'RMSE' for regression.
8. **border_count**: Number of splits for numerical features.
9. **feature_border_type**: The type of binarization for numerical features. Options include 'Median' and 'Uniform'.
10. **per_float_feature_quantization**: Per-feature quantization settings.
11. **input_borders**: Initial borders for numeric features.
12. **output_borders**: Final borders for numeric features.
13. **fold_permutation_block**: Size of fold permutation blocks. Used for reducing the effect of permutations.
14. **od_pval**: P-value threshold for overfitting detection. Helps in early stopping to prevent overfitting.
15. **od_wait**: Number of iterations to wait for overfitting detection to occur.
16. **od_type**: Type of overfitting detection. Options include 'IncToDec' or 'Iter'.
17. **nan_mode**: Handling mode for NaN values. Options include 'Min', 'Max', or 'Forbidden'.
18. **counter_calc_method**: Method of calculating counters for categorical features.
19. **leaf_estimation_iterations**: Number of iterations for leaf estimation.
20. **leaf_estimation_method**: Method for leaf estimation. Options include 'Newton' and 'Gradient'.
21. **thread_count**: Number of threads to use for training.
22. **random_seed**: Random seed for reproducibility.
23. **use_best_model**: Whether to use the best model during training (useful for early stopping).
24. **verbose**: Verbosity level for logging.
25. **logging_level**: Level of logging detail.
26. **metric_period**: Frequency of metric calculation and logging.
27. **ctr_leaf_count_limit**: Limit for the number of leaves when constructing categorical features.
28. **store_all_simple_ctr**: Whether to store all simple CTR values.
29. **max_ctr_complexity**: Maximum complexity of CTR combinations.
30. **has_time**: Indicator of time features in the dataset.
31. **allow_const_label**: Allow constant target labels.
32. **classes_count**: Number of classes in the dataset for classification.
33. **class_weights**: Class weights for balancing the dataset.
34. **auto_class_weights**: Automatic calculation of class weights.
35. **one_hot_max_size**: Maximum size for using one-hot encoding for categorical features.
36. **random_strength**: Strength of the randomness. Controls the level of randomness when selecting splits.
37. **name**: Custom name for the model.
38. **ignored_features**: List of features to be ignored during training.
39. **train_dir**: Directory to store training logs and snapshots.
40. **custom_loss**: Custom loss function.
41. **custom_metric**: Custom evaluation metric.
42. **eval_metric**: Metric for evaluating the model performance.
43. **bagging_temperature**: Temperature parameter for Bayesian bagging.
44. **save_snapshot**: Whether to save a snapshot of the training progress.
45. **snapshot_file**: File to store the training snapshot.
46. **snapshot_interval**: Interval for saving snapshots.
47. **fold_len_multiplier**: Multiplier for fold length.
48. **used_ram_limit**: Limit for the amount of RAM used during training.
49. **gpu_ram_part**: Fraction of GPU RAM to be used during training.
50. **allow_writing_files**: Allow writing of files during training.
51. **final_ctr_computation_mode**: Mode for final CTR computation.
52. **approx_on_full_history**: Approximate history usage for prediction.
53. **boosting_type**: Boosting type. Options include 'Ordered' and 'Plain'.
54. **simple_ctr**: Simple CTR settings.
55. **combinations_ctr**: Combinations CTR settings.
56. **per_feature_ctr**: Per-feature CTR settings.
57. **task_type**: Task type. Options include 'CPU' and 'GPU'.
58. **device_config**: Configuration for devices used in training.
59. **devices**: List of devices to be used for training.
60. **bootstrap_type**: Type of bootstrap. Options include 'Bayesian', 'Bernoulli', etc.
61. **subsample**: Subsample ratio for the dataset.
62. **sampling_unit**: Sampling unit type.
63. **dev_score_calc_obj_block_size**: Block size for score calculation.
64. **max_depth**: Maximum depth of the tree.
65. **n_estimators**: Number of trees (iterations).
66. **num_boost_round**: Number of boosting rounds.
67. **num_trees**: Number of trees to grow.
68. **colsample_bylevel**: Subsample ratio of columns for each level.
69. **random_state**: Random state for reproducibility.
70. **reg_lambda**: Regularization parameter.
71. **objective**: Objective function for training.
72. **eta**: Learning rate.
73. **max_bin**: Maximum number of bins for features.
74. **scale_pos_weight**: Scaling factor for positive class.
75. **gpu_cat_features_storage**: Storage type for categorical features on GPU.
76. **data_partition**: Data partitioning type.
77. **metadata**: Metadata information.
78. **early_stopping_rounds**: Number of rounds for early stopping.
79. **cat_features**: List of categorical features.
80. **grow_policy**: Policy for growing trees.
81. **min_data_in_leaf**: Minimum data points in a leaf.
82. **min_child_samples**: Minimum samples required in a leaf.
83. **max_leaves**: Maximum number of leaves.
84. **num_leaves**: Number of leaves in the tree.
85. **score_function**: Score function for selecting the best splits.
86. **leaf_estimation_backtracking**: Backtracking method for leaf estimation.
87. **ctr_history_unit**: Unit for CTR history.
88. **monotone_constraints**: Constraints for monotonicity of features.
89. **feature_weights**: Weights for each feature.
90. **penalties_coefficient**: Coefficient for penalties.
91. **first_feature_use_penalties**: Penalties for the first use of features.
92. **model_shrink_rate**: Shrink rate for the model.
93. **model_shrink_mode**: Shrink mode for the model.
94. **langevin**: Enable Langevin boosting.
95. **diffusion_temperature**: Temperature parameter for diffusion.
96. **posterior_sampling**: Enable posterior sampling.
97. **boost_from_average**: Enable boosting from the average.
98. **text_features**: List of text features.
99. **tokenizers**: Tokenizers for text processing.
100. **dictionaries**: Dictionaries for text processing.
101. **feature_calcers**: Calculators for feature processing.
102. **text_processing**: Text processing settings.
103. **fixed_binary_splits**: Enable fixed binary splits.


The range of value each hyper-parameter takes is as following:
1. **iterations**: Number of boosting iterations.
   - Range: Positive integer (e.g., 100 to 10000)
2. **learning_rate**: The rate at which the model learns.
   - Range: (0.01 to 1.0)
3. **depth**: Depth of the tree.
   - Range: Positive integer (e.g., 1 to 16)
4. **l2_leaf_reg**: L2 regularization term on weights.
   - Range: Positive float (e.g., 0 to 10)
5. **model_size_reg**: Regularization for the model size.
   - Range: Positive float
6. **rsm**: Random subspace method.
   - Range: (0.0 to 1.0)
7. **loss_function**: Loss function used to evaluate model performance.
   - Options: 'Logloss', 'CrossEntropy', 'RMSE', 'MAE', etc.
8. **border_count**: Number of splits for numerical features.
   - Range: Positive integer (e.g., 1 to 255)
9. **feature_border_type**: The type of binarization for numerical features.
   - Options: 'Median', 'Uniform', 'UniformAndQuantiles', 'MaxLogSum'
10. **per_float_feature_quantization**: Per-feature quantization settings.
   - Options: List of quantization settings
11. **input_borders**: Initial borders for numeric features.
   - Options: List of float values
12. **output_borders**: Final borders for numeric features.
   - Options: List of float values
13. **fold_permutation_block**: Size of fold permutation blocks.
   - Range: Positive integer
14. **od_pval**: P-value threshold for overfitting detection.
   - Range: (0.0 to 1.0)
15. **od_wait**: Number of iterations to wait for overfitting detection.
   - Range: Positive integer
16. **od_type**: Type of overfitting detection.
   - Options: 'IncToDec', 'Iter'
17. **nan_mode**: Handling mode for NaN values.
   - Options: 'Min', 'Max', 'Forbidden'
18. **counter_calc_method**: Method of calculating counters for categorical features.
   - Options: 'Full', 'SkipTest'
19. **leaf_estimation_iterations**: Number of iterations for leaf estimation.
   - Range: Positive integer
20. **leaf_estimation_method**: Method for leaf estimation.
   - Options: 'Newton', 'Gradient'
21. **thread_count**: Number of threads to use for training.
   - Range: Positive integer
22. **random_seed**: Random seed for reproducibility.
   - Range: Positive integer
23. **use_best_model**: Whether to use the best model during training.
   - Options: True, False
24. **verbose**: Verbosity level for logging.
   - Range: Positive integer
25. **logging_level**: Level of logging detail.
   - Options: 'Silent', 'Verbose', 'Info'
26. **metric_period**: Frequency of metric calculation and logging.
   - Range: Positive integer
27. **ctr_leaf_count_limit**: Limit for the number of leaves when constructing categorical features.
   - Range: Positive integer
28. **store_all_simple_ctr**: Whether to store all simple CTR values.
   - Options: True, False
29. **max_ctr_complexity**: Maximum complexity of CTR combinations.
   - Range: Positive integer
30. **has_time**: Indicator of time features in the dataset.
   - Options: True, False
31. **allow_const_label**: Allow constant target labels.
   - Options: True, False
32. **classes_count**: Number of classes in the dataset for classification.
   - Range: Positive integer
33. **class_weights**: Class weights for balancing the dataset.
   - Options: List of float values
34. **auto_class_weights**: Automatic calculation of class weights.
   - Options: 'Balanced', None
35. **one_hot_max_size**: Maximum size for using one-hot encoding for categorical features.
   - Range: Positive integer
36. **random_strength**: Strength of the randomness.
   - Range: Positive float
37. **name**: Custom name for the model.
   - Options: String
38. **ignored_features**: List of features to be ignored during training.
   - Options: List of integer indices
39. **train_dir**: Directory to store training logs and snapshots.
   - Options: String (directory path)
40. **custom_loss**: Custom loss function.
   - Options: List of strings
41. **custom_metric**: Custom evaluation metric.
   - Options: List of strings
42. **eval_metric**: Metric for evaluating the model performance.
   - Options: String
43. **bagging_temperature**: Temperature parameter for Bayesian bagging.
   - Range: Positive float
44. **save_snapshot**: Whether to save a snapshot of the training progress.
   - Options: True, False
45. **snapshot_file**: File to store the training snapshot.
   - Options: String (file path)
46. **snapshot_interval**: Interval for saving snapshots.
   - Range: Positive integer
47. **fold_len_multiplier**: Multiplier for fold length.
   - Range: Positive float
48. **used_ram_limit**: Limit for the amount of RAM used during training.
   - Range: String (e.g., '2Gb', '2048Mb')
49. **gpu_ram_part**: Fraction of GPU RAM to be used during training.
   - Range: (0.0 to 1.0)
50. **allow_writing_files**: Allow writing of files during training.
   - Options: True, False
51. **final_ctr_computation_mode**: Mode for final CTR computation.
   - Options: 'Default', 'SkipTest'
52. **approx_on_full_history**: Approximate history usage for prediction.
   - Options: True, False
53. **boosting_type**: Boosting type.
   - Options: 'Ordered', 'Plain'
54. **simple_ctr**: Simple CTR settings.
   - Options: List of CTR description objects
55. **combinations_ctr**: Combinations CTR settings.
   - Options: List of CTR description objects
56. **per_feature_ctr**: Per-feature CTR settings.
   - Options: List of CTR description objects
57. **task_type**: Task type.
   - Options: 'CPU', 'GPU'
58. **device_config**: Configuration for devices used in training.
   - Options: String
59. **devices**: List of devices to be used for training.
   - Options: List of strings
60. **bootstrap_type**: Type of bootstrap.
   - Options: 'Bayesian', 'Bernoulli', 'MVS'
61. **subsample**: Subsample ratio for the dataset.
   - Range: (0.0 to 1.0)
62. **sampling_unit**: Sampling unit type.
   - Options: 'Object', 'Group'
63. **dev_score_calc_obj_block_size**: Block size for score calculation.
   - Range: Positive integer
64. **max_depth**: Maximum depth of the tree.
   - Range: Positive integer
65. **n_estimators**: Number of trees (iterations).
   - Range: Positive integer
66. **num_boost_round**: Number of boosting rounds.
   - Range: Positive integer
67. **num_trees**: Number of trees to grow.
   - Range: Positive integer
68. **colsample_bylevel**: Subsample ratio of columns for each level.
   - Range: (0.0 to 1.0)
69. **random_state**: Random state for reproducibility.
   - Range: Positive integer
70. **reg_lambda**: Regularization parameter.
   - Range: Positive float
71. **objective**: Objective function for training.
   - Options: String
72. **eta**: Learning rate.
   - Range: (0.01 to 1.0)
73. **max_bin**: Maximum number of bins for features.
   - Range: Positive integer
74. **scale_pos_weight**: Scaling factor for positive class.
   - Range: Positive float
75. **gpu_cat_features_storage**: Storage type for categorical features on GPU.
   - Options: 'GpuRam', 'CpuPinnedMemory'
76. **data_partition**: Data partitioning type.
   - Options: 'Feature', 'Doc'
77. **metadata**: Metadata information.
   - Options: Dictionary
78. **early_stopping_rounds**: Number of rounds for early stopping.
   - Range: Positive integer
79. **cat_features**: List of categorical features.
   - Options: List of integer indices
80. **grow_policy**: Policy for growing trees.
   - Options: 'SymmetricTree', 'Depthwise', 'Lossguide'
81. **min_data_in_leaf**: Minimum data points in a leaf.
   - Range: Positive integer
82. **min_child_samples**: Minimum samples required in a leaf.
   - Range: Positive integer
83. **max_leaves**: Maximum number of leaves.
   - Range: Positive integer
84. **num_leaves**: Number of leaves in the tree.
   - Range: Positive integer
85. **score_function**: Score function for selecting the best splits.
    - Options: 'Cosine', 'L2', 'NewtonCosine', 'NewtonL2'
86. **leaf_estimation_backtracking**: Backtracking method for leaf estimation.
    - Options: 'No', 'AnyImprovement'
87. **ctr_history_unit**: Unit for CTR history.
    - Options: 'Log', 'Sum'
88. **monotone_constraints**: Constraints for monotonicity of features.
    - Options: Dictionary of feature indices and constraints
89. **feature_weights**: Weights for each feature.
    - Range: List of positive float values
90. **penalties_coefficient**: Coefficient for penalties.
    - Range: Positive float
91. **first_feature_use_penalties**: Penalties for the first use of features.
    - Range: Positive float
92. **model_shrink_rate**: Shrink rate for the model.
    - Range: (0.0 to 1.0)
93. **model_shrink_mode**: Shrink mode for the model.
    - Options: 'Constant', 'Decreasing'
94. **langevin**: Enable Langevin boosting.
    - Options: True, False
95. **diffusion_temperature**: Temperature parameter for diffusion.
    - Range: Positive float
96. **posterior_sampling**: Enable posterior sampling.
    - Options: True, False
97. **boost_from_average**: Enable boosting from the average.
    - Options: True, False
98. **text_features**: List of text features.
    - Options: List of feature indices
99. **tokenizers**: Tokenizers for text processing.
    - Options: List of tokenizers
100. **dictionaries**: Dictionaries for text processing.
    - Options: List of dictionaries
101. **feature_calcers**: Calculators for feature processing.
    - Options: List of feature calculators
102. **text_processing**: Text processing settings.
    - Options: List of text processing settings
103. **fixed_binary_splits**: Enable fixed binary splits.
    - Options: True, False
 
Among these parameters, the following are the most influencial ones:
Absolutely, here are some of the most important and impactful parameters you should consider when using `CatBoostClassifier`:

1. **iterations**: Number of boosting iterations.
   - This controls how many times the algorithm will pass through the data to improve the model. Typical values are between 100 and 1000.

2. **learning_rate**: The rate at which the model learns.
   - A smaller value (e.g., 0.01 to 0.1) generally leads to better performance, though training will be slower.

3. **depth**: Depth of the tree.
   - Determines how many layers the trees will have. Typical values range from 4 to 10.

4. **l2_leaf_reg**: L2 regularization term on weights.
   - Helps to prevent overfitting. Values typically range from 1 to 10.

5. **rsm**: Random subspace method.
   - Fraction of features to be used for fitting each tree. Values typically range from 0.5 to 1.0.

6. **loss_function**: Loss function used to evaluate model performance.
   - Common options are 'Logloss' for classification tasks and 'RMSE' for regression tasks.

7. **eval_metric**: Metric for evaluating the model performance.
   - It’s important to set this to a metric that aligns with your specific objective, such as 'Accuracy' or 'AUC'.

8. **od_type**: Type of overfitting detection.
   - Options include 'IncToDec' or 'Iter', and it helps in early stopping to prevent overfitting.

9. **od_wait**: Number of iterations to wait for overfitting detection.
   - This controls the patience of the early stopping mechanism. Typical values are between 20 and 50.

10. **random_strength**: Strength of the randomness.
    - Controls the level of randomness when selecting splits, which can help with overfitting. Typical values range from 1 to 10.

11. **bagging_temperature**: Temperature parameter for Bayesian bagging.
    - Influences the random subsampling of the data, with typical values around 0.5 to 1.0.

12. **depth**: Maximum depth of the tree.
    - Controls the maximum number of splits from the root to the leaf node. Typical values range from 4 to 10.
 
