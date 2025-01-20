# -------------------------------------------------------------------------------------------------------
#                           Training catBoost model via purged cross-validation
# -------------------------------------------------------------------------------------------------------

import numpy as np
import pandas as pd
import os, gc
from sklearn.model_selection import KFold
from catboost import CatBoostRegressor, Pool

def weighted_MAE(y_true, y_pred, weight):
    """
    Compute the weighted Mean Absolute Error.
    
    Parameters:
    y_true (array-like): True target values.
    y_pred (array-like): Predicted target values.
    weight (array-like): Weights for each sample.
    
    Returns:
    float: Weighted Mean Absolute Error.
    """
    return np.sum(weight * np.abs(y_true - y_pred)) / np.sum(weight)

def purged_cv(X, y, weights, X_test, 
              catboost_params, cat_features, 
              early_stopping_rounds=50, n_splits=5, purge_ratio=0.1, 
              save_models=True, save_predictions_path=None, train_on_GPU=False):
    """
    Purged Cross-Validation for time series data, with options to save models and generate predictions.

    Parameters:
    X (DataFrame): Feature data.
    y (Series): Target data.
    weights (Series): Weights for the samples.
    X_test (DataFrame): Test feature data for inference.
    catboost_params (dict): Optimized hyperparameters for the model.
    cat_features (list): List of categorical feature indices.
    n_splits (int): Number of folds for cross-validation.
    purge_ratio (float): Percentage of the most recent data to be purged from each training set.
    save_models (bool): Whether to save the best model for each fold.
    save_predictions_path (str): Path to save fold predictions as CSV files.

    Returns:
    dict: Contains average MAE, models, and predictions.
    """
    kf = KFold(n_splits=n_splits, shuffle=False)
    
    total_weighted_mae = 0
    fold = 0
    fold_predictions = []  # To store predictions for averaging
    models = []  # To store models if saving is enabled
    test_predictions = []  # To store test set predictions
    results = {}

    if save_predictions_path and not os.path.exists(save_predictions_path):
        os.makedirs(save_predictions_path)

    for train_index, valid_index in kf.split(X):
        fold += 1
        print(f"Processing Fold {fold}")
        
        # Split data into train and validation sets
        X_train, X_valid = X.iloc[train_index], X.iloc[valid_index]
        y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]
        weights_train, weights_valid = weights.iloc[train_index], weights.iloc[valid_index]
        
        # Purge the most recent data from the training set
        purge_size = int(len(X_train) * purge_ratio)
        X_train = X_train[:-purge_size]
        y_train = y_train[:-purge_size]
        weights_train = weights_train[:-purge_size]

        # Create Pool for CatBoost with categorical features
        train_pool = Pool(X_train, y_train, cat_features=cat_features, weight=weights_train)
        valid_pool = Pool(X_valid, y_valid, cat_features=cat_features, weight=weights_valid)

        if train_on_GPU:
            catboost_params.update({
                'gpu_cat_features_storage': 'GpuRam',#'CpuPinnedMemory',
                'task_type': "GPU",
                'devices': '0'
            })

        # Train the model on the purged data using the optimized parameters
        model = CatBoostRegressor(**catboost_params)
        model.fit(train_pool, 
                  eval_set=valid_pool,
                  early_stopping_rounds=early_stopping_rounds,
                  verbose=False)
        
        # Save the best model if required
        if save_models:
            models.append(model)
        
        # Predictions and evaluate using the custom weighted MAE
        y_pred = model.predict(valid_pool)
        fold_weighted_mae = weighted_MAE(y_valid, y_pred, weights_valid)
        total_weighted_mae += fold_weighted_mae

        print(f"Fold {fold} Weighted MAE: {fold_weighted_mae}")
        
        # Store the fold predictions for later use (averaging across folds)
        fold_predictions.append(y_pred)

        # Predict on the test data for inference
        if X_test is not None:
            test_pool = Pool(X_test, cat_features=cat_features)
            test_pred = model.predict(test_pool)
            test_predictions.append(test_pred)

            # Save fold predictions to disk if the path is provided
            if save_predictions_path:
                fold_pred_df = pd.DataFrame({
                    'fold': [fold] * len(test_pred),
                    f'predictions_{fold}': test_pred
                })
                fold_pred_df.to_csv(os.path.join(save_predictions_path, f"fold_{fold}_predictions.csv"), index=False)
        
        gc.collect()
    
    # Save results in a dictionary
    results['models'] = models
    results['avg_weighted_mae'] = total_weighted_mae / n_splits
    if X_test is not None:
        results['test_prediction'] = test_predictions
    print(f"Average Weighted MAE: {results['avg_weighted_mae']}")
          
    return results


# Example usage after performing hyperparameter optimization with Optuna
def run_purged_cv_optimized_model(X_train: pd.DataFrame, y_train: pd.Series, 
                                  weights_train: pd.Series, 
                                  X_test: pd.DataFrame = None,
                                  cat_features = None,
                                  n_splits = 5,
                                  save_models = True,
                                  early_stopping_rounds = 50,
                                  catboost_params = None, 
                                  purge_ratio = 0.1, 
                                  save_predictions_path = None,
                                  train_on_GPU = False):
    """
    Run Purged Cross-Validation on the optimized model.
    
    Parameters:
    X_train (DataFrame): Features.
    y_train (Series): Target variable.
    weights_train (Series): Sample weights.
    X_test (DataFrame): Test feature data for inference.
    cat_features (list): List of categorical feature indices.
    catboost_params (dict): Optimized hyperparameters for the model.
    n_splits (int): Number of folds for cross-validation.
    purge_ratio (float): The fraction of data to purge from the training set.
    save_predictions_path (str): Path to save predictions during training.
    train_on_GPU (bool): Whether to train on GPU.

    Returns:
    dict: Contains average MAE, models, and predictions.
    """
    if catboost_params is None:
        catboost_params = {
            'learning_rate': 0.03,
            'depth': 2, 
            'iterations': 100,
            }
    if train_on_GPU:
        catboost_params.update({
            'loss_function': 'RMSE',
            
            'random_seed': 42,
            'verbose': 0
        })
    else:
        catboost_params.update({
            'loss_function': 'MAE',

            'random_seed': 42,
            'verbose': 0
        })

    # Using Purged Cross-Validation
    results = purged_cv(X_train, y_train, weights_train, 
                        X_test, catboost_params, cat_features,
                        early_stopping_rounds=early_stopping_rounds,
                        n_splits=n_splits, 
                        save_models=save_models,
                        purge_ratio=purge_ratio, 
                        save_predictions_path=save_predictions_path,
                        train_on_GPU=train_on_GPU)
    
    return results


# -----------------------------------------
# Example
# -----------------------------------------
results = run_purged_cv_optimized_model(X_train               = X_train, 
                                        y_train               = y_train, 
                                        weights_train         = weights_train, 
                                        X_test                = X_test,
                                        cat_features          = cat_cols,
                                        n_splits              = 5,
                                        save_models           = True,
                                        early_stopping_rounds = 50,
                                        catboost_params       = None, 
                                        purge_ratio           = 0.1, 
                                        save_predictions_path = Path(r"\Work\cbt"),
                                        train_on_GPU          = True
                                       )
