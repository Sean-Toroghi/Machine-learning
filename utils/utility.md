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

