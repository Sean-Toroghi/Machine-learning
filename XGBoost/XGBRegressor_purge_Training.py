# ------------------------------------------------------------------------------
#   Employ purge training method to train XGBRegressor on time series dataset
# ------------------------------------------------------------------------------


 import os, gc
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import xgboost as xgb
from xgboost import XGBRegressor

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

def purged_cv(X, y, weights, X_test, xgb_parameters, n_splits=5, purge_ratio=0.1, save_models=True, save_predictions_path=None):
    """
    Purged Cross-Validation for time series data, with options to save models and generate predictions.

    Parameters:
    X (DataFrame): Feature data.
    y (Series): Target data.
    weights (Series): Weights for the samples.
    X_test (DataFrame): Test feature data for inference.
    xgb_parameters (dict): Optimized hyperparameters for the model.
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

        # Train the model on the purged data using the optimized parameters
        model = XGBRegressor(**xgb_parameters)
        model.fit(X_train, y_train, sample_weight=weights_train, eval_set=[(X_valid, y_valid)], verbose=False)
        
        # Save the best model if required
        if save_models:
            models.append(model)
        
        # Predictions and evaluate using the custom weighted MAE
        y_pred = model.predict(X_valid)
        fold_weighted_mae = weighted_MAE(y_valid, y_pred, weights_valid)
        total_weighted_mae += fold_weighted_mae

        print(f"Fold {fold} Weighted MAE: {fold_weighted_mae}")
        
        # Store the fold predictions for later use (averaging across folds)
        fold_predictions.append(y_pred)
         
        # Predict on the test data for inference, if provided
        if X_test is not None:
            test_pred = model.predict(X_test)
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
    
    print(f"Average Weighted MAE: {results['avg_weighted_mae']}")
    
    # If X_test is provided, return the averaged predictions
    if X_test is not None:
        # Averaging the predictions from each fold
        results['avg_fold_predictions'] = np.mean(fold_predictions, axis=0)
        results['avg_test_predictions'] = np.mean(test_predictions, axis=0)

        # Optionally save the averaged test predictions
        if save_predictions_path:
            avg_test_pred_df = pd.DataFrame({
                'avg_test_predictions': results['avg_test_predictions']
            })
            avg_test_pred_df.to_csv(f"{save_predictions_path}_avg_test_predictions.csv", index=False)

    return results


# Example usage after performing hyperparameter optimization with Optuna
def run_purged_cv_optimized_model(X_train: pd.DataFrame, y_train: pd.Series, 
                                  weights_train: pd.Series, 
                                  X_test: pd.DataFrame = None,
                                  n_splits = 5,
                                  save_models = True,
                                  xgb_parameters = None, 
                                  gpu = False,
                                  include_categorical = True,
                                  purge_ratio = 0.1, 
                                  save_predictions_path = None):
    """
    Run Purged Cross-Validation on the optimized model.
    
    Parameters:
    X_train (DataFrame): Features.
    y_train (Series): Target variable.
    weights_train (Series): Sample weights.
    X_test (DataFrame): Test feature data for inference.
    xgb_parameters (dict): Optimized hyperparameters for the model.
    n_splits (int): Number of folds for cross-validation.
    purge_ratio (float): The fraction of data to purge from the training set.
    save_predictions_path (str): Path to save predictions during training.
    
    Returns:
    dict: Contains average MAE, models, and predictions.
    """
    if xgb_parameters is None:
        xgb_parameters = {
            'learning_rate': 0.07,
            'max_depth': 7, 
            'n_estimators': 1800,
            'objective': 'reg:squarederror',
            'random_state': 42,
        }

    if include_categorical:
        xgb_parameters['enable_categorical'] = True
        
    if gpu:
        xgb_parameters.update({
            'device': 'cuda',
            'tree_method': 'gpu_hist',
            'updater': 'grow_gpu_hist',
            'updater_seq': 'grow_gpu_hist'
        })

    # Using Purged Cross-Validation
    results = purged_cv(X_train, y_train, weights_train, 
                        X_test, xgb_parameters, 
                        n_splits=n_splits, 
                        save_models=save_models,
                        purge_ratio=purge_ratio, 
                        save_predictions_path=save_predictions_path)
    
    return results

# -------------------------------------------------------------------------------------
# Example, running the training
# -------------------------------------------------------------------------------------

X_train = train[num_cols + cat_cols]
y_train = train[target]
weights_train = train[weights]
X_test = test[num_cols + cat_cols]
del train, test
gc.collect()
print(X_train.shape, y_train.shape, weights_train.shape, X_test.shape)

# -------------------------------------------------------------------------------------
xgb_tuned_parameters = {
    # phase 1
    'learning_rate'   : 0.0286879283986566,
    'max_depth'       : 13 ,
    'n_estimators'    : 1858 ,
    # phase 2
    'subsample'       : 0.885763464818004 ,
    'colsample_bytree': 0.8661799767968346,
    # phase 3
    'reg_lambda'      : 1.0067040820839281, 
    'reg_alpha'       : 0.02410856857385563, 
    'gamma'           : 4.339052419845749, 
    'min_child_weight': 3,
    }

Result_tuned_8folds = run_purged_cv_optimized_model(
    X_train             = X_train,
    y_train             = y_train, 
    weights_train       = weights_train,
    X_test              = X_test,
    n_splits            = 5,
    save_models         = True,
    xgb_parameters      = xgb_tuned_parameters,
    gpu                 = True,
    include_categorical = True,
    purge_ratio         = 0.1, 
    make_predictions    = True,
    save_predictions_path="/working_directory"
    )

print(f"Average Weighted MAE:   {avg_weighted_mae}")
print(f"Fold Predictions size: {len(avg_fold_predictions)}")
print(f"Test Predictions size: {len(avg_test_predictions)}")

 
