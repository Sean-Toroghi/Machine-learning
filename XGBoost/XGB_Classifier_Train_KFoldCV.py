import os, gc
from re import A
import numpy as np
import pandas as pd
from typing import Optional
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from sklearn.utils import class_weight
from xgboost import XGBClassifier


def xgb_kfold_cv(X, y, X_test, xgb_parameters, n_splits=5, 
                 save_models=True, save_predictions_path=None, APPLY_WEIGHT = False):
    """
    K-Fold Cross-Validation for training XGB classification,
     with options to save models and generate predictions.

    Parameters:
    X (DataFrame): Feature data.
    y (Series): Target data.
    X_test (DataFrame): Test feature data for inference.
    xgb_parameters (dict): Optimized hyperparameters for the model.
    n_splits (int): Number of folds for cross-validation.
    save_models (bool): Whether to save the best model for each fold.
    save_predictions_path (str): Path to save fold predictions as CSV files.

    Returns:
    dict: Contains average AUC, models, and predictions.
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    total_auc = 0
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

        if APPLY_WEIGHT:
          classes_weights = class_weight.compute_sample_weight(
              class_weight='balanced',
              y=y_train
          )

        # Train the model on the full training data using the optimized parameters
        model = XGBClassifier(**xgb_parameters)
        if APPLY_WEIGHT:
          model.fit(X_train, y_train, sample_weight=classes_weights, eval_set=[(X_valid, y_valid)], verbose=False)
        else:
          model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)

        # Save the best model if required
        if save_models:
            models.append(model)

        # Predictions and evaluation using regular AUC
        y_pred = model.predict_proba(X_valid)[:, 1]
        fold_auc = roc_auc_score(y_valid, y_pred)
        total_auc += fold_auc

        print(f"Fold {fold} AUC: {fold_auc}")

        # Store the fold predictions for later use (averaging across folds)
        fold_predictions.append(y_pred)

        # Predict on the test data for inference
        if X_test is not None:
            test_pred = model.predict_proba(X_test)[:, 1]
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
    results['avg_auc'] = total_auc / n_splits
    if X_test is not None:
        results['test_prediction'] = test_predictions
    print(f"Average AUC: {results['avg_auc']}")

    return results


def run_xgb_classifier_kfold_cv(X_train: pd.DataFrame, y_train: pd.Series,
                                X_test: Optional[pd.DataFrame] = None,
                                n_splits = 5,
                                save_models = True,
                                xgb_parameters = None,
                                save_predictions_path = None,
                                train_on_GPU = False,
                                APPLY_WEIGHT = False):
    """
    Run K-Fold Cross-Validation on XGB classifier model.

    Parameters:
    X_train (DataFrame): Features.
    y_train (Series): Target variable.
    X_test (DataFrame): Test feature data for inference.
    xgb_parameters (dict): Optimized hyperparameters for the model.
    n_splits (int): Number of folds for cross-validation.
    save_predictions_path (str): Path to save predictions during training.
    train_on_GPU (bool): Whether to train on GPU.
    APPLY_WEIGHT (bool): Whether to apply class weights.

    Returns:
    dict: Contains average AUC, models, and predictions.
    """
    if xgb_parameters is None:
        xgb_parameters = {
            # Tier 1
            'learning_rate': 0.03,
            'max_depth': 7,                  # Maximum depth of trees
            'n_estimators': 1800,            # Number of trees
            # Tier 2
            'min_child_weight': 5,           # Minimum number of samples in a leaf node
            'gamma': 1e-6,                   # Minimum loss reduction
            'subsample': 0.8,                # Fraction of samples for training each tree
            'colsample_bytree': 0.8,         # Fraction of features for training each tree

            'objective': 'binary:logistic',  # Binary classification with logistic regression
            #'objective': 'multi:softprob'    #Multiclass classification
            'eval_metric': 'auc',

            'random_state': 42,
        }

    if train_on_GPU:
        xgb_parameters.update({
            'device': 'cuda',
            'tree_method': 'gpu_hist',
            'updater': 'grow_gpu_hist',
        })
    else:
        xgb_parameters.update({
        'n_jobs': -1,
        })

    # Using Regular K-Fold Cross-Validation
    results = xgb_kfold_cv(X_train, y_train, X_test,
                           xgb_parameters,
                           n_splits=n_splits,
                           save_models=save_models,
                           save_predictions_path=save_predictions_path,
                           APPLY_WEIGHT=APPLY_WEIGHT)

    return results
# --------------------------------------------
#                   Example
# --------------------------------------------
results = run_xgb_classifier_kfold_cv(
    X_train,
    y_train,
    X_test,
    n_splits = 5,
    save_models = True,
    xgb_parameters = None,
    save_predictions_path = root/'xgb_v00',
    train_on_GPU = False,
    APPLY_WEIGHT = True)
