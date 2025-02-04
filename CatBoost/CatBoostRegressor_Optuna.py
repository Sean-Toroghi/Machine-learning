# -----------------------------------------------------------------------------------
#                   Perform hyper-parameter tuning in 3 phases
# -----------------------------------------------------------------------------------


import optuna
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

# -----------------------------------------------------------------------------------
def weighted_MAE(y_true, y_pred, weight):
    return np.sum(weight * np.abs(y_true - y_pred)) / np.sum(weight)
# -----------------------------------------------------------------------------------
def objective_catboost(trial, X_train, y_train, X_valid,
                  y_valid, weights_train,
                  weights_valid, cat_cols, train_on_GPU):
    
    params = {
        ## Set 1 - Hyperparameters with most impact
        #'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.05, log=True),
        #'depth': trial.suggest_int('depth', 3, 14),
        #'iterations': trial.suggest_int('iterations', 1000, 3000),

        ## Set 2 - Hyperparameters for fine-tuning
        #'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1.0, 10.0, log=True),
        #'model_size_reg': trial.suggest_float('model_size_reg', 0.1, 1.0, log=True),
        #'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
        #'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        #'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.5, 1.0),
        #
        # Set 3 - Hyperparameters with less impact
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 1, 10),
        'max_bin': trial.suggest_int('max_bin', 100, 500),
        'leaf_estimation_iterations': trial.suggest_int('leaf_estimation_iterations', 1, 5),
        'leaf_estimation_method': trial.suggest_categorical('leaf_estimation_method', ['Newton', 'Gradient']),
        'feature_border_type': trial.suggest_categorical('feature_border_type', ['GreedyLogSum', 'Uniform', 'Median', 'MaxLogSum']),

        'random_seed': 42,
        'verbose': 0
    }

    if train_on_GPU:
        params.update({
            'task_type': 'GPU',
            'devices': '0',
            'loss_function': 'RMSE'
        })
    else:
        params.update({
            'loss_function': 'MAE'
        })

    params.update(
        {
            # Tier 1
            'learning_rate': 0.04645075433302456, 
            'depth': 14, 
            'iterations': 1984,
            # Tier 2
            'l2_leaf_reg': 1.3652039509690967, 
            'model_size_reg': 0.10067837027178779, 
            'bagging_temperature': 0.0014818761943015124
        }
    )
    # Create Pool for CatBoost (needed to specify categorical features)
    train_pool = Pool(X_train, label=y_train, weight=weights_train, cat_features=cat_cols)
    valid_pool = Pool(X_valid, label=y_valid, weight=weights_valid, cat_features=cat_cols)

    model = CatBoostRegressor(**params,)
    model.fit(train_pool,
              eval_set=valid_pool,
              early_stopping_rounds=50,
              verbose=False)

    y_pred = model.predict(valid_pool)
    weighted_mae = weighted_MAE(y_valid, y_pred, weights_valid)
    print(f"Weighted MAE: {weighted_mae}")
    return weighted_mae
# -----------------------------------------------------------------------------------
def run_optim_study_catboost(
    train_df,
    num_cols, cat_cols,
    n_trials,
    APPLY_STANDARDIZATION, APPLY_SMOTE,
    train_on_GPU=False):

    featureCols = num_cols + cat_cols
    target_col = 'sales'
    weight_col = 'weight'

    # Ensure weights are included in the split to prevent mismatch
    X_train, X_valid, y_train, y_valid = train_test_split(
        train_df[featureCols], train_df[target_col],
        test_size=0.2, random_state=42
    )
    weights_train, weights_valid = X_train[weight_col], X_valid[weight_col]

    # Remove weights from feature sets after splitting
    X_train = X_train.drop(columns=[weight_col, target_col])
    X_valid = X_valid.drop(columns=[weight_col, target_col])

    if APPLY_SMOTE:
        print("Applying SMOTE")

        # Concatenate X_train and weights_train to handle them together during resampling
        X_train_with_weights = X_train.copy()
        X_train_with_weights['weight'] = weights_train

        # Apply SMOTE to the concatenated data
        smote = SMOTE(random_state=42)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train_with_weights, y_train)

        # Extract the resampled weights from the last column after resampling
        weights_train_resampled = X_train_resampled['weight']
        X_train_resampled = X_train_resampled.drop(columns=['weight'])

        X_train, y_train, weights_train = X_train_resampled, y_train_resampled, weights_train_resampled

    if APPLY_STANDARDIZATION:
        print("Applying Standardization")
        scaler = StandardScaler()
        features = [col for col in num_cols if col not in ['sales', 'weight']]

        X_train[features] = scaler.fit_transform(X_train[features].copy())
        X_valid[features] = scaler.transform(X_valid[features].copy())

    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective_catboost(trial,
                                                    X_train, y_train, X_valid, y_valid,
                                                    weights_train, weights_valid,
                                                    cat_cols, train_on_GPU),
                n_trials=n_trials)

    best_params = study.best_params
    print(f"Best parameters for CatBoostRegressor: {best_params}")
    return best_params

# Example

param_tie_3 = run_optim_study_catboost(
    train,
    num_cols = num_cols + [weights] + [target],
    cat_cols = cat_cols,
    n_trials = 15,
    APPLY_STANDARDIZATION = False, 
    APPLY_SMOTE = False,
    train_on_GPU=True)
print(param_tie_3)
