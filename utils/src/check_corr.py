import pandas as pd

def check_corr(df, return_corr_matrix = False):
  '''
  Compute correlation matrix and check if any feature is correlated with 
    another feautre with corr score larger than 0.95 and returns the corr matrix
    if return_corr_matrix is True.
    Also prints the columns to drop.

  Input: 
    df: dataframe
    return_corr_matrix: boolean
  '''

  # check if df is dataframe
  if not isinstance(df, pd.DataFrame):
    raise TypeError("Input must be a pandas DataFrame.")

  if df.empty:
    raise ValueError("Input DataFrame is empty.")


  print("Compute correlation.")
  corr_matrix = df.corr().abs()
  upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
  to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.95)]
  print(f"Checked feature corr in dataset {i} Columns to drop: {to_drop}\n\n")
  if return_corr_matrix:
    return corr_matrix
  else:
    return None
