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

