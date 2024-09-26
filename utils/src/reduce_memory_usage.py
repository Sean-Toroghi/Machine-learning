# save memory
import numpy as np
import pandas as pd

def reduce_mem_usage(df, verbose=True):
	'''
	Input: 
		dataframe
		verbose: Boolean - generate a report at the end
	Output:
		downcasting version of dataframe
	'''
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2
    
    for col in df.columns:
        col_type = df[col].dtypes
        if str(col_type) in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type).startswith('int'):
                if c_min >= np.iinfo(np.int8).min and c_max <= np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min >= np.iinfo(np.int16).min and c_max <= np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min >= np.iinfo(np.int32).min and c_max <= np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                # No need for an additional check here since int64 is the default
                # only downcasting to int16 and int32 if applicable
                else:
                    print(f"Error in column {col}: unable to downcast integer type")
            elif str(col_type).startswith('float'):
                if c_min >= np.finfo(np.float16).min and c_max <= np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min >= np.finfo(np.float32).min and c_max <= np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                # No need for an additional check here since float64 is the default
                # So we only consider downcasting to float16 and float32
                else:
                    print(f"Error in column {col}: unable to downcast float type")

    end_mem = df.memory_usage().sum() / 1024**2
    
    if verbose:
        print(f"Memory usage of DataFrame is {start_mem:.3f} MB")
        print(f"Memory usage after optimization is: {end_mem:.3f} MB")
        print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
        
    return df