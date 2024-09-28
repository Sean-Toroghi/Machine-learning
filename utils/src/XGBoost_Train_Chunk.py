%%time

# Train XGboost model with large dataset by chunking the data into smaller pieces and iteratively go over each during the trainig phase
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor

# Initialize the XGBRegressor
model = XGBRegressor(
    objective='reg:squarederror',
    eval_metric='rmse',
    learning_rate=0.01,
    max_depth=6,
    min_child_weight=1,
    subsample=0.8,
    colsample_bytree=0.2,
    gamma=0,
    reg_lambda=1,  # L2 regularization
    reg_alpha=0,   # L1 regularization
    n_estimators=100,
    random_state=42,
    #n_jobs = -1,
    device= 'cuda'
)


# Train model
class TQDMProgress(callback.TrainingCallback):
    def __init__(self, total):
        self.pbar = tqdm(total=total)

    def after_iteration(self, model, epoch, evals_log):
        self.pbar.update(1)
        return False  # Continue training

    def __del__(self):
        self.pbar.close()
        
# divide the dataset into chuncks (chunk_size = samples in each round)   
# with 100, it requires 11GB ram
chunk_size = 100
chunk_i = 1
# Fit the model with tqdm progress
for start in range(0, len(train), chunk_size): 
    print(f"Chunk {chunk_i}")
    end = start + chunk_size
    model.fit(train[start:end],
              train_labels[start:end],
              callbacks=[TQDMProgress(total=100)] #Change this value based on n_estimators
             )
    chunk_i+=1
