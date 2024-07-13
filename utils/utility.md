# Machine learning

## Feature engineering

### Categorical features
- _convert to numerical values_ via `pd.get_dummies` or `sklearn.preprocessing.OneHotEncoder`
- _Frequency of each class_: convert categorical columns into their frequencies, which equates to the percentage of times each category appears within the given column.
  ```python
  df['cat_freq'] = df.groupby('categorical_feature')['categorical_feature'].transform('count') 
  df['cat_freq'] = df['cat_freq']/len(df)
  ```
- _Mean encoding (also called target encoding)_: transforms categorical columns into numerical columns based on the mean target variable. To avoid data leakage, we need to apply a regularization technique after the transformation.
  ```python
  from category_encoders.target_encoder import TargetEncoder
  encoder =  TargetEncoder()
  df['categorical_mean_encoded'] = encoder.fit_transform(df['categogircal_feature'], df['target'])
  ```
- 


# coding 
## print all elements in a list
```python
# option 1
with np.printoptions(threshold=np.inf):
    print(np.array(long_list))

# option 2
import sys
np.set_printoptions(threshold=sys.maxsize)
np.array(cols)

```

## replace header with column names and add the header as a row to the dataset
```python
original_cols = df.columns.tolist()
df.columns = all_cols
df_original_cols = pd.DataFrame([original_cols], columns=all_cols)
df = pd.concat([df, df_original_cols], ignore_index=True)
```

