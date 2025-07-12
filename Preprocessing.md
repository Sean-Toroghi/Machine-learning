# Pre-processing 
- Outlier detection
- Feature selection
- Missing values
__References__
[^1]: Book - Python Feature Engineering Cookbook - Second Edition By Soledad Gall
[^2]: Book - Feature Engineering for Machine Learning By Alice Zheng and Amanda Casari
[^3]: Book - Python Data Cleaning and Preparation Best Practices by Maria Zervou


---
## Outlier detection


---
## Feature selection
In machine learning, feature selection is an important part of the preprocessing in a ML-pipeline. There are several feature selection methods, among which are:
- correlation coefficient based metods
- information theory based methods
- statistical (test) based methods
- model-based feature selection methods
- wrapper methods


### Correlation coefficient based methods
This approach first computes correlation b/w each feature and target variable. In the next step it computes the iner-feature orelation, in which we compute the correlation b/w ach pair of features. Some of the most common correlation methods are:
1. __Pearson correlation__: The Pearson correlation coefficient is calculated using the covariance of the two variables divided by the product of their standard deviations. As the result, in inherently standardizes the values. This method computes existance of a linear correlation.
2. __Spearman's rank correlation__ ($\rho$): measures nomotonic and non-linear relationship btw variables. It also is used when the data ir ordinal or does not follow the normal distribution. This method is more robust to outliers than Pearson.
3. __Kendall's Tau__ ($\tau$): measures strength and direction of monotonic relationships btw two ranked variables. It is an alternative approach to Spearman' rank method. This method is used when we have a small dataset, and provides a more interpretable probability of concodance indication.
4. Distance correlation: is a more general measure of dependency btw two random variables. It can detect any relationship (linear, non-linear, monotonic, non-monotonic). However, it is computationally more intensive, compare with the Peason or Spearman.
5. Maximal information coefficiant:  measures the strength of any functional relationship (linear or non-linear) between two variables. It's designed to capture a wide range of dependencies. Similar to distance correlation, it is used to discover complex / non-linear relationship.


__Note__: threshold tuning is an important part of this approach. To derive an optimal threshold, it is needed to  analyze different threshold, and also employ cross-validation. 

### Information theory-based methods
These methods quantify how much information one vaiable provides about another variable. Some of the most common methods are:
1. Mutual information: measures the amount of information obtained about one random variable by observing another. It quantifies the reduction in uncertainty about the target variable, given the value of a feature. It can capture non-linear relationships. It is a versatile method that works both with discrete and continuous variables. sklearn: `mutual_info_regression` and `mutual_info_classif`.
2. Information gain: often used in tree-based method, in which it quantifies the reduction in entropy (or impurity) of the target variable when split the data based on a feature.

### Statistical-based methods
The statistical-based approach evaluates features individually w.r.t. target variable. Two common methods are
1. ANOVA test: measures linear dependency between a continuous target and a continuous (or categorical, when used for classification) feature. It calculates the F-statistic, which essentially compares the variance between group means to the variance within groups. __sklean: `f_regression`__
2. Chi-square test: measures independence between two categorical variables. So it is used when both fature and target variable are categorical. It compares observed frequencies in a contingency table to expected frequencies if the variables were independent. __sklean: `chi2`__

### Model-based approach
The model-based methods incorporate feature selection within the model. Among these methods are Lasso regression (L1), Ridge regression (L2), and three-based methods  (e.g., Random Forest, Gradient Boosting Machines). Example `RandomForestRegressor.feature_importances_`

### Wrapper apprach
While computationaly expensive, they can find optimal feature subsets for a given model. This method is used when the number features is limited, since it is computationally expensive. Among them the most common methods are:
1. Recursive feature elimination: this method iteratively trains a model (e.g., a linear regression, SVM, or tree-based model), gets the importance/coefficients, removes the least important feature(s), and repeats until the desired number of features is reached. It often combines with cross-validation method. __sklean: `RFECV`__
2. Forward selection: starts with no features and iteratively adds the feature that best improves the model's performance until no further improvement is seen or a set number of features is reached.
3. Backward selection:starts with all features and iteratively removes the feature that least hurts the model's performance until no further improvement/deterioration is seen.
4. Stepwise selection: is a combination of forward and backward, adding and removing features at each step. 
