

References  
- [link 1: Deep Learning for Survival Analysis](https://humboldt-wi.github.io/blog/research/information_systems_1920/group2_survivalanalysis/)
- [link 2](https://github.com/robi56/Survival-Analysis-using-Deep-Learning)
- Wiegrebe, Simon, Philipp Kopper, Raphael Sonabend, Bernd Bischl, and Andreas Bender. "Deep Learning for Survival Analysis: A Review." *Artificial Intelligence Review* 57, no. 3 (February 2024): 157-173. [link](https://doi.org/10.1007/s10462-023-10681-3).
- [paper: A Survey on Multi-Task Learning - 2021](https://arxiv.org/abs/1707.08114)
Sure! Here's the provided BibTeX entry converted to Chicago format:



---
# Deep learning approach
To categorize DL models, three metrics could be use: 
1. model class, based on which type of statistical survival technique is used to form the DL model.
2. loss function, and combination of loss-functions.
3. parameterization, that defines which part of a model is parametrized by a NN method. 

__Cox-based approach__

This approach employs DL to model Cox regression, parameterizing the log-risk function (hazard rate) by a NN and minimize the Cox-loss (neg. log of the partial likelihood of the Cox model).

Examples:
- DeepSurv by Katzman (2018)
-  Cox-Time by Kvamme et al. (2019) - extension of DeepSurv
-  NN-DeepSurv b yTong and Zhao (2022) - extension of DeepSurv

__Discrete-time approach__

this approach considers the time to be discrete and in most cases employs classification techniques with binary event indicater for each descret timestamp. The standard loss function for this approach is negative log-likelihood and it is typical for the discrete hazard to be parameterized by the NN. 

This approach is much more heterogeneous in terms of loss function and architecture, compare wit hCox-based approach.

Examples:
- DeepHit by Lee et al. (2018)
- Dynamic-DeepHit by Lee et al. (2019)
-  TransformerJM  by Lin and Luo (2022)
-  Nnet-survival  by Gensheimer and Narasimhan (2019)
-  Tho2022  by Thorsen-Meyer (2022)
-  DRSA by Ren (2019)

__Parametric approach__

__PEM‑based approach__

# Cox PH model
The Cox PH model provides expression for hazard at point of time _t_ with a given specification of a set of explanatory variables. According to the Cox model, hazard at time _t_ is a function of two parameters: baseline hazard, and the exponential _e_ to the linear form of the sum of independent variables $\beta_i X_i$. the baseline hazard is a function of _t_, while does not involve X's. The exponential part is a function of X's, but does not involve _t_. Here X's are time-independent. 

$$Cox-PH: h(t, X) = h_0(t) e^{\sum (\beta_i X_i)}$$

Cox PH model assumes the hazard ratio that compares any two specifications of predictors is constant over time. 
A Cox model with time-dependent X's is called the __extended Cox model__. 

# Evaluate proportional hazard (PH assumption in Cox-PH model)

Among different methods for evaluating proportional hazard (PH), three famous ones are: 1. graphical,2. goodness-of-fit, and 3. time-dependent variables. 
- __Graphical techniques__: in short, the graphical method compares two graphs over time for two groups and if they are independent over time, they should show such a behavior on the plotted graph. Some of the graphical techniques are:
  - Schoenfeld Residuals Plot: The residuals should be independent of time if the PH assumption holds. A plot of Schoenfeld residuals against time should show no pattern or trend.
  - Log(-Log(Survival)) Plot: plotting the log(-log(survival)) against log(time), if the PH assumption is valid, the plot should show parallel lines for different levels of the predictor.
  - Kaplan-Meier Curves:for categorical predictors, Kaplan-Meier curves can be used. If the PH assumption holds, the survival curves for different groups should be roughly parallel.
  - Time-Dependent Covariates: Including time-dependent covariates in the model can help test the PH assumption. If the coefficients of these covariates are significant, it indicates a violation of the PH assumption.
  - Observed w/ predictor: comparing observed vs predictor survival curve.
- __Goodness-of-fit__: for a large sample of Z or chi-square statistics, based on p-values derived from standard normal statistics for each variable, if it is not significant, it indicates the PH assumption is satisfied. One downside of the GOF method could be it is too global and may not detect a specific violation of PH assumption.
- __time-dependent variable__: generating a new feature by multiplying time by a time-independent variable, creates a time-dependent feature. Now if the coefficient of this new feature be significant, the PH assumption is violated for the original feature. 
 

## Stratified Cox model
The Stratified Cox model, is a modification to Cox model that uses stratification to control predictors that do not satisfy PH assumption. Stratification could be applied to one or more variables. A stratified method could be with no-interaction, or run with interaction.  The stratified Cox model assumes that there is no interaction between the stratification variable and other covariates. This means that the effect of the stratification variable is not modified by the other covariates

__Single-variable stratification__

In this method, if a variable is time-dependent, it is stratified in the model, while the other time-independent variables are included.
  
__General form (k-variable stratification)__

Given a dataset with _k_ variables not satisfying the PH assumption and _p_ variables that satisfy the PH assumption, we perform the stratification as following:
- Forming categories of time-dependent variables.
- Forming combinations of the categories driven in the previous step.
- Each combination is a strata.

## Extention of Cox model (time-dependent variables)

__Time-dependent vs time-independent variables__
- time-dependent: variables that their values change over time, such as age. _Internal_ variables are also time-dependent, such as exposure.
- time-independent: variables that their values do not change over time, such as race. However, a time-independent variable could be converted into a time-dependent variable. One method is to multiply it by a time variable of time, such as _(t-1)_ or _g(t)_ (a function of time). One example of a time function is _heaviside_ function, defined as the following:

$$g(t) = \begin{cases}
1 & t \leq t_0 \\
0 & t < t_0
\end{cases}
$$
Employ heaviside function $E g(t)$ will be $E$ for $t \leq t_0$, and _0_ for all other cases.

__Ancillary variables__ is another type of variable for which  its value changes primarily because of “external” characteristics of the environ ment that may affect  several individuals simultaneously. An example of this type of variables is air polution.

__General formula for extended Cox PH model__
The formula for the extended Cox-PH model consists of two terms, one associated with time-dependendt variables and the other time-independent variables: 
$$Extended\ Cox-PH: h(t, X) = h_0(t) e^{\sum (\beta_i X_i) + \sum (\gama_j Z_j)}$$





























