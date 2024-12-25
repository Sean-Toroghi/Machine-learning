<h1>Polars - summary</h1>

Polars package is a fast data processing and manupulation tool built on top of Rust. While, in general, it resembles methods in Pandas, for most cases it uses different names, approache, and structure for doring the same task. In many cases, I find it beneficial to have a quick reminder while coding in Polars. Not only it saves a good time, but also it helps me to pratice more efficiently. I tried to keep the balance between  breadth and depth, and structure it for quick referencing. 

__table of content__
- [Expression](#expression)

__References__
- [Polars - offcial](https://docs.pola.rs/py-polars/html/reference/)


---
# <a id='expression'>Expressions</a>

An expression is a tree of operations that describe how to construct one or more Series (an array of values with the same data type). Expressions in Polar is used for selecting a column, creating a column, perform aggregation, and fitering rows. They are just a description, in tree format, that executed when they are passed as arguments to functions. 

The expression class in Polar (`polars.Expr`) has over 350 methods. These methods are categorized under namespaces in the expression, such as string (`Expr.str`), temporal values (`Expr.dt`), and categorical variables (`Expr.cat`).  It is neither practical, nor beneficial to put them all here. I create a list and desription of the fundamental methods. 

Expression are applied by passing them to a dataframe of lazyframe. 

## Experession methods
### Selecting columns - `df.select()`

This method is used to select one or more columns. Any column not mentioned in the selection are dropped from the output. Also, for any name not in the original dataframe, a new column is created. 

__Example__ select two column, and create a new column.
```python
df_1.select(
  pl.col('name'),
  pl.col('orrders'),
  pl.col('weight') / 1000, 'weight_kg')
```

### Create new column - `df.with_columns()`
This method is used to create new column/s, wither from existing ones, or scratch. 

__Example__
- create two new features
- create a column from scratch and use `alias()` to give the new column a name
- create second column frm exisitng column
```python
df_1.with_columns(
  pl.lit(True).alias('Boolean_new_column'),  # create a boolean column with initial value True 
  column_name_2 = pl.col('name').str.ends_with('some_string')
)
```

### Filtering rows - `df.filter()`

### Aggregation - `df.groupby(...).agg(...)`

### Sorting rows - `df.sort()`

### Filtering - 



### Perform mathematical transformation
 

### Missing values

### Apply smoothing

### Selecting a value

### Summarizing statstical values

