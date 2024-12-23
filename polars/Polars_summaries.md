<h1>Polars - summary</h1>

Polars package is a fast data processing and manupulation tool built on top of Rust. While, in general, it resembles methods in Pandas, for most cases it uses different names, approache, and structure for doring the same task. In many cases, I find it beneficial to have a quick reminder while coding in Polars. Not only it saves a good time, but also it helps me to pratice more efficiently. I tried to keep the balance between  breadth and depth, and structure it for quick referencing. 

__table of content__
- [Expression](#expression)


---
# <a id='expression'>Expressions</a>

Expressions in Polar is used for selecting a column, creating a column, perform aggregation, and fitering rows. The expression class in Polar (`polars.Expr`) has over 350 methods. These methods are categorized under namespaces in the expression, such as string (`Expr.str`), temporal values (`Expr.dt`), and categorical variables (`Expr.cat`).  It is neither practical, nor beneficial to put them all here. I create a list and desription of the fundamental methods. 

Expression are applied by passing them to a dataframe of lazyframe. 

## Selecting columns - `df.select()`

This method is used to select one or more columns. Any column not mentioned in the selection are dropped from the output. Also, for any name not in the original dataframe, a new column is created. 

__Example__ select two column, and create a new column.
```python
df_1.select(
  pl.col('name'),
  pl.col('orrders'),
  pl.col('weight') / 1000, 'weight_kg')
```
