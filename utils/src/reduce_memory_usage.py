import numpy as np
import pandas as pd

def reduce_mem_usage(df, EXTREME=False, verbose=True):
    """
    Reduce memory usage of a DataFrame by downcasting numeric types.
    
    Parameters:
    df: DataFrame to optimize
    EXTREME: If True, allows downcasting to float16
    verbose: If True, prints memory usage details
    
    Returns:
    Optimized DataFrame
    """
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2

    # Use a dictionary to map dtypes to their respective min/max values
    for col in df.select_dtypes(include=numerics).columns:
        col_type = df[col].dtype

        if str(col_type).startswith('int'):
            c_min = df[col].min()
            c_max = df[col].max()
            if c_min >= np.iinfo(np.int8).min and c_max <= np.iinfo(np.int8).max:
                df[col] = df[col].astype(np.int8)
            elif c_min >= np.iinfo(np.int16).min and c_max <= np.iinfo(np.int16).max:
                df[col] = df[col].astype(np.int16)
            elif c_min >= np.iinfo(np.int32).min and c_max <= np.iinfo(np.int32).max:
                df[col] = df[col].astype(np.int32)

        elif str(col_type).startswith('float'):
            c_min = df[col].min()
            c_max = df[col].max()
            if EXTREME and (c_min >= np.finfo(np.float16).min and c_max <= np.finfo(np.float16).max):
                df[col] = df[col].astype(np.float16)
            elif c_min >= np.finfo(np.float32).min and c_max <= np.finfo(np.float32).max:
                df[col] = df[col].astype(np.float32)

    end_mem = df.memory_usage().sum() / 1024**2

    if verbose:
        print(f"Memory usage of DataFrame is {start_mem:.3f} MB")
        print(f"Memory usage after optimization is: {end_mem:.3f} MB")
        print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
        
    return df
