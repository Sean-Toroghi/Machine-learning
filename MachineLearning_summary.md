<h1>Machine learning - quick reference</h1>

# Steps involve in a machine learning project

A machine learning prject, in its general form, goes through the following phases:
0. define problem
1. collecting and aquiring data
2. preprocess and clean data
   - EDA
   - Feature engineering 
4. select algorithm apprpriate for the task in hand, and process of selecting best algorithm among all cadidates
  - develop a baseline to evalue model performance in the next step (either a simple algorithm with low cost and/or heuristic-based (i.e., rules-based) approach)
  - training process and optimizing the model to gain highest performance, including fine-tuneining, hyperparameter tuning, experiment tracking. 
4. evaluate model and examine its generalization capability

# Step 0 - define problem
Perhaps define and translate a problem into ML problem is the most critical point in any ML project. It requires to clearly define problem, scope, budget, requirement, cost, potential for future expantion, and feasibility. Sometime, a ruled-based model can deliver the same results as of a complex and sophisticated model generate. Also, have a clear understanding of problem requires, in many case, gather domain knowledge. This later will help when performing feature engineering and also choosing ML algorithms. Finally, define boundaries (scopre, budget, cost, time, ...) influence the choice of approach throughout the project, from data aquisition to model update.

# Step 1 - collecting and acquiring data

__References__
- [Making Scence of Data - By Glenn J. Myatt, Wayne P. Johnson](https://learning.oreilly.com/library/view/making-sense-of/9781118422106/)
- [Hands-On Data Preprocessing in Python - By Roy Jafari](https://learning.oreilly.com/library/view/hands-on-data-preprocessing/9781801072137/)
- [Hands-On Exploratory Data Analysis with Python - By Suresh Kumar Mukhiya, Usman Ahmed](https://learning.oreilly.com/library/view/hands-on-exploratory-data/9781789537253/)

There are many options available for data acquisition in the context of ML, such as access to proprietary data, public datasets, web scraping, purchasing data from vendors, and creating synthetic data.

# Step 2 - data preprocessing and feature engineering

## 2.1 - EDA
Explanatory data analysis is the first step in data processing. In a general form, it consists of 
- gain a high-level overview of distribution of data: mean, s.d., max, min, ... for continous, and count, number of unique categories, ... for categorical variables
- find any potential flaws, such as missing vlaues, skewness, duplicates, and outliers


To interpret the resutls of EDA, it is important to have some domain knwoledge. Simply looking at statistical results may not be enought

__Tools__
- Pandas and Polars functions for EDA
- [ydata-profiling]()
- Create custom function to generate a report
- Visualization (python packages (Matplotlib, Seaborn, Plotly), Softwares (Tableau))
  
## 2.2 - Feature engineering
The goal of feature engineering is to make modifications to the dataset to ensure its compatibility with ML algorithms and fix any flaws or incompleteness in data. 
- imputation
- duplicate data
- standardizing data
- preprocessing (categoricl encoding, binning, feature selection, ...)

  
### Perform imputation
While some ML algorithms can inherently handle missing values, some other cannot. In most cases, it is much more safer to handle missing values at the initial stage of data preprocessing to avoid any future interuption in the pipeline. 

If the sample size be large enough and removing rows/clolumn with missing values does not negatively effect model (for example rows with missing values represent a target class and deleting them removes a target class form dataset), we can simply delete rows with missing values. Also, this decision can be made based on the rate of missing values in a row.

Beside removing samples with missing values, we can perform imputation. Imputation methods are categorized into two main categories:

| Technique         | Pros                                                                                       | Cons                                                                                                                            |
|-------------------|--------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------|
| Mean/median/mode/prior value, next available value/average or past n values,...  | Simple to implement                                                                        | Might not account for outliers compared to tree-based methods Not as suitable for categorical variables                         |
| Tree-based or other ML models | Can capture more underlying patterns Suitable for both numerical and categorical variables | Adds a level of complexity during data preprocessing Model needs to be retrained if the underlying distribution of data changes |

__Data leakage and imputation__

Imputation, if not properly perform, can lead to data leakage. For example, using mean to impute a feature needs to be done after splitting data into train/val/test and seperately to each set.


### Address duplicate data
Python and SQL are equiped with methods to remove duplicate variables, such as coverting a list to set in Python, or UNIQUE in SQL.

### Standardizing data

Data standardization includes: handling outliers, scaling features, and  data type consistency.

__Outliers__

To address outlier issue in dataset, we have several options
- define a threshold (e.g. any value above/below 3s.d. over mean) and remove outliers from sample
- winsorizing: replacing outliers with less extreme vlaues
- perform logarithmic scale transform

Removing outliers requires having domain knwoledge, as in some casesit leads to lowering the capability of model to generalize. 

__Scale features__

Scale of features effect many ML algorithms. There are two main methods for scaling features, normalization and standardization. It is important to consider effect of outlier on scaling. For example, a feature with extreme outliers creates missleading when we pick min and max to perform scaling
- standardization (z-score normalization) data range $\[-1,1\]$: by transform data to have mean = 0 and s.d. = 1
- normalization data range $\[0,1\]$ by computing the following ratio $\frac{x_i-\mu}{x_{max} - x_{min}}$

__Data type consistency__

In some cases there is a missmatch between feature and its data type. For example, a numerical feature can be saved as string in dataframe. Perform a data-type survey during the preprocessing phase ensures the constency between stored data type and expected data type for a feature.

### Data preprocessing
Data preprocessing involves changing data to the format that can be interpreted by a ML algorithm. 

#### Categorical encoding

While some ML algotrithm can handle categorical features in their original form, such as CatBoost, many ML algorithm only can handle numerical vlaues. Hense, those algorithms require the categorical features to be converted to numerical. Some of the techniques for encoding categorical data are as follow:
- One-hot encoding: similar to get_dummy function, this class (from scikit-learn) generates binary columns for eah unique class. In addition, it can handle unseen categories. IT is appropriate for nominal data. Similar to get_dummy, in case of high cardinality it is required to group together categories to avoid generating too many new features.
- Factorize: this function (from Numpy) assign unique interger to each unique category and returns an array of integers aliong with the uqniue categories. It is similar to label encoding, but also provides mapping. As the result, it is appropriate for ordinal data.
- get_ummy method in Pandas: creates a binary column for each category. This method is appropriate for nominal categores with no intrinsic order. The downside of this method is it creates sparse dataset. Furthermore, this method is not efficient and negatively effect model performacnce if applies to categorical features with high cardinality (curse of dimensionality). In such a case, we need to first downsize the number of categories by grouping unique categories and then apply it. 
- label encoding: convert categorical values into integer. This method assign unioque number to each unique category. This is appropriate for _ordinal_ data.
- target encoding (mean encoding):
- binary encoding
- frequency encoding
- ordinal encoding
- hashing encoding
- leave-one-out encoding
- CatBoost encoding



__Binning numerical values__

Binning is used to reduce the cardinality and improves model generalization capability. It creates bins and allocate numerical values into them. A downside of binning is that it introduces hard edges into the meanings of the bins.

__Feature selection__

In some cases, features may have high collinearity with each other. Furthermore, some features may capture a high proportion of the same information as another feature. Remvoing the features that are highly correlated with other features, reduce chance of overfitting and improve speed of training process.

Some techniques used for feature selection are
- correlation matrix
- feature importance generated by some tree-based models

Also dimentionality reduction method can be used to reduce the dimentionality of data.

### Class imbalance

Class imbalance can negatively effect the performance of a ML method. The following traditional strategies can be employed to address class imbalance:

- Undersampling reduces the size of the majority class to match the minority class. This helps balance the dataset but may result in the loss of valuable majority class examples.
- Oversampling increases the size of the minority class to match the majority class by duplicating existing examples. This improves balance but can lead to model overfitting on the repeated data examples.
- Synthetic Minority Over-sampling TEchnique (SMOTE) is similar to oversampling, except that it generates synthetic examples instead of simply duplicating existing ones. It does this by generating new instances based on examples that are similar in feature space.

## Training model
Model training is an iterative proces and includes the following steps
- define ML task
- select the most usitable ML algorithms
- train model

### Define ML task

### Model selection


### Train model

__Hyperparameter tuning__

__Define loss function__

__Optimize model__

__Track model performance__

## Model evaluation


### Classification metrics

__Precision__

__Recall__

__Accuracy__

__F1-score__

__AUC-ROC__

### Regression metrics

__MAE__

__MSE__

__RMSE__

__$R^2$__

### Clustering metrics

__Silhouette coefficient__

__Calinski-Harabasz Index__

### Ranking metrics

__Mean reciprocal rank (MRR)

__Precision at K__

__Normalized discounted cumulative gain (NDCG)__


### Trade-off in evaluation metrics

### Offline-evaluation

### Model versioning
