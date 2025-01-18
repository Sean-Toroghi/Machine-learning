# Machine learning encoding methods


## `factrorize`
This method transforms unique categories into numerical values. It is a function in Pandas. It assignes -1 to `Nan` values. 

__(-) Cons__
- not suitable to be used directly in machine learning pipeline provided by sklearn
- no built-in function to transform labels back to original values
- 
```python
import pandas as pd

# Sample data
data = pd.Series(['cat', 'dog', 'mouse', 'dog', 'cat'])

# Factorize the data
encoded_data, unique_values = pd.factorize(data)

print(encoded_data)
print(unique_values)
# >>> [0 1 2 1 0]
# >>> Index(['cat', 'dog', 'mouse'], dtype='object')
```

---
## `LabelEncoder`
This method transforms unique categories into numerical values. It is a part of sklearn ecosystem, and can be integrated into a sklean pipeline. It handles encoding and decoding tasks.

__(-) cons__
- two step (fit-transform) approach

```python
from sklearn.preprocessing import LabelEncoder

# Sample data
data = ['cat', 'dog', 'mouse', 'dog', 'cat']

# Initialize LabelEncoder
le = LabelEncoder()

# Fit and transform the data
encoded_data = le.fit_transform(data)

print(encoded_data)
# >>> [0 1 2 1 0]
```
