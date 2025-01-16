import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import xgboost as xgb
from xgboost import XGBRegressor
# ------------------------------------------------------------
# Employ purge training method to trian XGBRegressor on time series dataset
# ------------------------------------------------------------

def weighted_MAE(y_true, y_pred, weight):
    return np.sum(weight * np.abs(y_true - y_pred)) / np.sum(weight)

def purged_cv(X, y, weights, X_test, xgb_parameters, n_folds=5, purge_ratio=0.1, save_models=True, make_predictions=False):
    """
    Purged Cross-Validation for time series data, with options to save models and generate predictions.

    Parameters:
    X (DataFrame): Feature data.
    y (Series): Target data.
    weights (Series): Weights for the samples.
    X_test (DataFrame): Test feature data for inference.
    optimized_params (dict): Optimized hyperparameters for the model.
    n_splits (int): Number of folds for cross-validation.
    purge_ratio (float): Percentage of the most recent data to be purged from each training set.
    save_models (bool): Whether to save the best model for each fold.
    make_predictions (bool): Whether to return the predictions for each fold.

    Returns:
    float: Average Weighted MAE from the cross-validation.
    list: (Optional) List of predictions from each fold (if make_predictions=True).
    list: (Optional) List of test predictions (if make_predictions=True).
    """
    kf = KFold(n_splits=n_folds, shuffle=False)
    
    total_weighted_mae = 0
    fold = 0
    fold_predictions = []  # To store predictions for averaging
    models = []  # To store models if saving is enabled
    test_predictions = []  # To store test set predictions
    
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
        model = XGBRegressor(
            **xgb_parameters  # Pass optimized parameters here
        )

        model.fit(X_train, y_train, sample_weight=weights_train, 
                  eval_set=[(X_valid, y_valid)], 
                  verbose=False)
        
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
        
        # Predict on the test data for inference
        test_pred = model.predict(X_test)
        test_predictions.append(test_pred)
    
    # Average Weighted MAE over all folds
    avg_weighted_mae = total_weighted_mae / n_folds
    print(f"Average Weighted MAE: {avg_weighted_mae}")
    
    # If make_predictions is True, return the averaged predictions
    if make_predictions:
        # Averaging the predictions from each fold
        avg_fold_predictions = np.mean(fold_predictions, axis=0)
        avg_test_predictions = np.mean(test_predictions, axis=0)
        return avg_weighted_mae, avg_fold_predictions, avg_test_predictions
 
    return avg_weighted_mae , None, None


# Example usage after performing hyperparameter optimization with Optuna
def run_purged_cv_optimized_model(X_train: pd.DataFrame, y_train: pd.Series, 
                                  weights_train: pd.Series, 
                                  X_test: pd.DataFrame,
                                  xgb_parameters = None, 
                                  gpu = False,
                                  include_categorical = True,
                                  n_folds: int =5, 
                                  purge_ratio=0.1, 
                                  make_predictions=False):
    """
    Run Purged Cross-Validation on the optimized model.
    
    Parameters:
    X_train (DataFrame): Features.
    y_train (Series): Target variable.
    weights_train (Series): Sample weights.
    X_test (DataFrame): Test feature data for inference.
    optimized_params (dict): Optimized hyperparameters for the model.
    n_splits (int): Number of folds for cross-validation.
    purge_ratio (float): The fraction of data to purge from the training set.
    make_predictions (bool): Whether to return the predictions or not.
    
    Returns:
    float: The average weighted MAE across all folds.
    list: (Optional) The averaged predictions (if make_predictions=True).
    list: (Optional) The averaged test predictions (if make_predictions=True).
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
      xgb_parameters.update(
          {
          'device': 'cuda', 
          'tree_method': 'gpu_hist',
          
          "updater": "grow_gpu_hist",
          "updater_seq": "grow_gpu_hist"
            }
          )        
    
    # Using Purged Cross-Validation
    avg_weighted_mae, avg_fold_predictions, avg_test_predictions = purged_cv(X_train, y_train, weights_train, 
                                                                             X_test, xgb_parameters, 
                                                                             n_folds=n_folds, 
                                                                             purge_ratio=purge_ratio, 
                                                                             make_predictions=make_predictions
                                                                              )
 
    return avg_weighted_mae, avg_fold_predictions, avg_test_predictions
 
 
 
 
 
 
 
 
 
 
# -------------------------------------------------------------------------------------
# Example running the training
# -------------------------------------------------------------------------------------

X_train = train[num_cols + cat_cols]
y_train = train[target]
weights_train = train[weights]
X_test = test[num_cols + cat_cols]
del train, test
gc.collect()
print(X_train.shape, y_train.shape, weights_train.shape, X_test.shape)

# -------------------------------------------------------------------------------------
avg_weighted_mae, avg_fold_predictions, avg_test_predictions = run_purged_cv_optimized_model(X_train, y_train, 
																							 weights_train,
                                                                                             X_test, 
                                                                                             xgb_parameters = None, 
                                                                                             gpu = False,
                                                                                             include_categorical = True,
                                                                                             n_folds=5, 
                                                                                             purge_ratio=0.1, 
                                                                                             make_predictions=True)

print(f"Average Weighted MAE:   {avg_weighted_mae}")
print(f"Fold Predictions size: {len(avg_fold_predictions)}")
print(f"Test Predictions size: {len(avg_test_predictions)}")

 
