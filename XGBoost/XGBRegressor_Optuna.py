import numpy as np
import pandas as pd
from xgboost import XGBRegressor, callback
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import optuna
# -----------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------
# Hyper-parameters tuning for XGBRegressor (sklean imeplementation) via optuna 
# By Sean Toroghi
# -----------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------
def weighted_MAE(y_true, y_pred, weight):
    return np.sum(weight * np.abs(y_true - y_pred)) / np.sum(weight)
# -----------------------------------------------------------------------------------
def objective_xgb(trial, X_train, y_train, X_valid, y_valid, weights_train, weights_valid, cat_cols):
    # add early stopping rounds
    es = callback.EarlyStopping(rounds=50,
                                        #min_delta=1e-3,
                                        #save_best=True,
                                        #maximize=False,
                                        #data_name="validation_0",
                                        #metric_name= 'rmse',
                                        )
    params = {
        # initial parameters to tune
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'max_depth': trial.suggest_int('max_depth', 4, 12),
        'n_estimators': trial.suggest_int('n_estimators', 500, 2000),
        
        ## Second layer parameters to tune: reduce overfitting
        #'subsample': trial.suggest_float('subsample', 0.5, 1.0, log=True),
        #'colsample_bytree': trial.suggest_float('colsample_bytree', 0.1, 1.0),
        
        ## fine-tune - regulaization parameters
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
 

    model = XGBRegressor(**params)
    model.fit(X_train, y_train, 
              sample_weight=weights_train, 
              eval_set=[(X_valid, y_valid)], 
              verbose=False)

    y_pred = model.predict(X_valid)
    weighted_mae = weighted_MAE(y_valid, y_pred, weights_valid)
    print(f"Weighted MAE: {weighted_mae}")
    return weighted_mae
# -----------------------------------------------------------------------------------
def run_optim_study_xgb(train_df, 
                        num_cols, cat_cols, 
                        n_trials, 
                        APPLY_STANDARDIZATION, APPLY_SMOTE):
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

    print(f"The shape of X_train is: {X_train.shape}")
    print(f"The shape of X_valid is: {X_valid.shape}")
    print(f"The shape of y_train is: {y_train.shape}")
    print(f"The shape of y_valid is: {y_valid.shape}")

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

        print(f"The shape of X_train after SMOTE is: {X_train_resampled.shape}")
        print(f"The shape of y_train after SMOTE is: {y_train_resampled.shape}")
        print(f"The shape of weights_train after resampling is: {weights_train_resampled.shape}")

        # Replace resampled data and weights for subsequent training
        X_train, y_train, weights_train = X_train_resampled, y_train_resampled, weights_train_resampled

    if APPLY_STANDARDIZATION:
        print("Applying Standardization")
        scaler = StandardScaler()
        features = [col for col in num_cols if col not in ['sales', 'weight']]
 
        X_train[features] = scaler.fit_transform(X_train[features].copy())
        X_valid[features] = scaler.transform(X_valid[features].copy())

    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective_xgb(trial, 
                                               X_train, y_train, X_valid, y_valid, 
                                               weights_train, weights_valid, 
                                               cat_cols), 
                   n_trials=n_trials)

    best_params = study.best_params
    print(f"Best parameters for XGBRegressor: {best_params}")
    return best_params
	
# ----------------------------------------------------------------------------------	
# Example for running the optimization
# Inputs
train = pd.DataFrame(...)
num_cols = [...]
cat_cols = [...]

best_params = run_optim_study_xgb(train, 
                                  num_cols , cat_cols, 
                                  n_trials=50, 
                                  APPLY_STANDARDIZATION=False, 
                                  APPLY_SMOTE=False)

print("Best hyperparameters:", best_params)	
