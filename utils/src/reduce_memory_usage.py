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
    
    # Check if input is a DataFrame
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    if df.empty:
        raise ValueError("Input DataFrame is empty.")

    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2

    def downcast_column(col):
        c_min = col.min()
        c_max = col.max()
        
        if pd.api.types.is_integer_dtype(col):
            if np.iinfo(np.int8).min <= c_min <= np.iinfo(np.int8).max:
                return col.astype(np.int8)
            elif np.iinfo(np.int16).min <= c_min <= np.iinfo(np.int16).max:
                return col.astype(np.int16)
            elif np.iinfo(np.int32).min <= c_min <= np.iinfo(np.int32).max:
                return col.astype(np.int32)
        elif pd.api.types.is_float_dtype(col):
            if EXTREME and np.finfo(np.float16).min <= c_min <= np.finfo(np.float16).max:
                return col.astype(np.float16)
            elif np.finfo(np.float32).min <= c_min <= np.finfo(np.float32).max:
                return col.astype(np.float32)
        
        return col  # return original if no downcasting is applied

    for col in df.select_dtypes(include=numerics).columns:
        try:
            df[col] = downcast_column(df[col])
        except Exception as e:
            print(f"Could not downcast column '{col}': {e}")

    end_mem = df.memory_usage().sum() / 1024**2

    if verbose:
        print(f"Memory usage of DataFrame is {start_mem:.3f} MB")
        print(f"Memory usage after optimization is: {end_mem:.3f} MB")
        print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
        
    return df
