# <a id = 'up'>Survival analysis</a>

__Table of contents__
- [Kaplan-Meier Survival Curves](#km)
- [Log-Rank Test](#logrank) 
- [Cox PH model](#coxph)
- [General Stratified Cox (SC) Model](#strcox)
- [Extention of Cox PH model](#coxextent)
- [Parametric survival model](#parametric)

__References__
- [link 1: Deep Learning for Survival Analysis](https://humboldt-wi.github.io/blog/research/information_systems_1920/group2_survivalanalysis/)
- [link 2](https://github.com/robi56/Survival-Analysis-using-Deep-Learning)
- Wiegrebe, Simon, Philipp Kopper, Raphael Sonabend, Bernd Bischl, and Andreas Bender. "Deep Learning for Survival Analysis: A Review." *Artificial Intelligence Review* 57, no. 3 (February 2024): 157-173. [link](https://doi.org/10.1007/s10462-023-10681-3).
- [paper: A Survey on Multi-Task Learning - 2021](https://arxiv.org/abs/1707.08114)
Sure! Here's the provided BibTeX entry converted to Chicago format:


---
---

# <a id = 'km'> [Kaplan-Meier Survival Curve](#up) </a>

The Kaplan-Meier Survival Curve is a non-parametric method for estimating survival function. It also provides a visual graph (curve) representing the probability of survival over time. 

Some of the key components of a KM approach are:
- Survival function: the Kaplan-Meier estimator calculates the survival function. It is the probability that an individual survives from the time of origin (e.g., diagnosis) to a specified time.
- Censored Data: this method accounts for censored data (it incorporates information from individuals whose event status is unknown at the end of the study).
- Step Function: the survival curve is a step function that changes only at the time of each event (e.g., death).
- Survival curve: a visual representation of the probability of survival over time.

__Kaplan-Meier Estimator Formula__: 

The Kaplan-Meier survival estimate at time $\( t \) : \[ \hat{S}(t) = \prod_{t_i \leq t} \left(1 - \frac{d_i}{n_i}\right) \]$

where:
- $\( t_i \)$ = Time at which the event occurred.
- $\( d_i \)$ = Number of events (e.g., deaths) at time $\( t_i \)$.
- $\( n_i \)$ = Number of individuals at risk just before time $\( t_i \)$.

Step to plot KM curve
1. collect survival times and event indicators.
2. employ the KM estimator to calculate the survival probabilities.
3. plot the estimated survival probabilities against time.
 
__Surivival curve__
- Survival Probability: The y-axis represents the estimated survival probability at each time point.
- Steps: The steps in the curve correspond to the times when events occurred. The curve drops at each event, reflecting the decrease in survival probability.
- Censoring is often indicated by tick marks on the curve.

__Pros and Cons__

- (+) KM curves effectively handle censored data.
- (+) The step function provides an easy-to-understand visualization of survival probabilities over time.
- (+) it does not assume a specific underlying distribution for survival times (non-parametric).
- (-) KM curves do not directly incorporate covariates. For that, models like the Cox proportional hazards model are used.
- (-) For comparing survival curves between groups, statistical tests like the log-rank test are used.

__Example (lifelines package)__

```python
import pandas as pd
from lifelines import KaplanMeierFitter
import matplotlib.pyplot as plt

# Data-sample
data = {
    'time': [5, 10, 15, 20, 25, 30],  # Survival times
    'status': [1, 0, 1, 0, 1, 1]  # Event indicators (1 = event, 0 = censored)
}
df = pd.DataFrame(data)

# Create Kaplan-Meier Fitter and fit the data
kmf = KaplanMeierFitter()
kmf.fit(df['time'], event_observed=df['status'])


# Plotting the survival curve
kmf.plot_survival_function()
plt.xlabel('Time')
plt.ylabel('Survival Probability')
plt.title('Kaplan-Meier Survival Curve')
plt.show()
```





---
# <a id = 'logrank'> [Log-Rank Test](#up) </a>

The log-rank test is a hypothesis test and is used to compare the survival distribution of two or more groups. It examines if there is a significant difference between the survival curves. 
- null-hyp.: there is no difference between the survival distribution
- alt-hyp.: there is a difference between the survival curve
- A low p-value (e.g., < 0.05) suggests a significant difference in survival distributions between the groups.
- log-rank test for large groups: it is approximately a chi-square test with _G-1_ degree of freedom (_G_ is number of groups).

__Example__
```python
import pandas as pd
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
import matplotlib.pyplot as plt

# dataset
data = {
    'time': [5, 6, 6, 6, 7, 10, 13, 16, 20, 25, 5, 6, 6, 8, 10, 12, 14, 18, 22, 30],
    'event': [1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1],
    'group': ['A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B']
}
df = pd.DataFrame(data)

# Fit Kaplan-Meier survival curves for each group
kmf_A = KaplanMeierFitter()
kmf_B = KaplanMeierFitter()
# Group A
kmf_A.fit(durations=df[df['group'] == 'A']['time'], event_observed=df[df['group'] == 'A']['event'], label='Group A')
# Group B
kmf_B.fit(durations=df[df['group'] == 'B']['time'], event_observed=df[df['group'] == 'B']['event'], label='Group B')

# Plot the survival curves
ax = kmf_A.plot_survival_function()
kmf_B.plot_survival_function(ax=ax)
plt.title('Kaplan-Meier Survival Curves')
plt.xlabel('Time')
plt.ylabel('Survival Probability')
plt.show()

# Perform Log-Rank Test
# Perform log-rank test
results = logrank_test(
    df[df['group'] == 'A']['time'], df[df['group'] == 'B']['time'],
    event_observed_A=df[df['group'] == 'A']['event'], event_observed_B=df[df['group'] == 'B']['event']
)

# Print results
print(f'Log-Rank Test p-value: {results.p_value}')
print(f'Test Statistic: {results.test_statistic}')
```


---

# <a id = 'coxph'> [Cox proportional hazard (PH) model](#up) </a>

Cox PH model is used to assess the effect of covariates on hazard function.  It is a statistical model that provides an expression for hazard at a point in time (_t_), with a given specification defined as a set of explanatory variables. According to the Cox PH model, hazard at time _t_ is a function of two parameters: 
1. baseline __hazard function__ represents the instantaneous risk of the event occurring at time _t_, given that the individual has survived up to time _t_. It is a function of _t_, while it does not involve X's.
2. the exponential _e_ to the linear form of the sum of independent variables $\beta_i X_i$.  The exponential part is a function of X's, but does not involve _t_. Here X's are time-independent.  The model estimates regression coefficients that quantify the effect of each covariate on the hazard rate.

 


$$Cox-PH: h(t, X) = h_0(t) e^{\sum (\beta_i X_i)}$$

**Proportional Hazards Assumption**

The Cox model assumes that the hazard ratios between groups are constant over time. This means that the effect of covariates on the hazard rate is multiplicative and does not change over time. A Cox model with time-dependent X's is called the __extended Cox model__. 



__Steps to run a Cox Model__

1. **Data Preparation**: Collect survival times, event indicators, and covariates.
2. **Fit the Model**:   fit the Cox model and estimate the regression coefficients.
3. **Check Assumptions**: Assess the proportional hazards assumption (e.g. diagnostic plots and tests).
4. **Interpret Results**: Interpret the regression coefficients and their statistical significance.


__Example__
```python
import pandas as pd
from lifelines import CoxPHFitter

# data
data = {
    'time': [5, 6, 6, 6, 7, 10, 13, 16, 20, 25],
    'event': [1, 1, 0, 1, 1, 1, 1, 1, 0, 1],
    'age': [50, 60, 45, 70, 55, 65, 40, 75, 60, 80],
    'treatment': [1, 1, 0, 0, 1, 1, 0, 0, 1, 1]
}
df = pd.DataFrame(data)

# Fit the Cox Model
# Create a CoxPHFitter object
cph = CoxPHFitter()
# Fit the model
cph.fit(df, duration_col='time', event_col='event')
# Print the summary of the model
cph.print_summary()

#>>> a p-value < 0.05 indicates the covariate is significant.
#>>> 'exp(coef)' represents the hazard ratio. For example, the value 1.03 indicates each additional amount(number) of that covariates increases the hazard by 3%

```


## Evaluate proportional hazard (PH assumption in Cox-PH model)

Among different methods for evaluating proportional hazard (PH), three famous ones are: 1. graphical,2. goodness-of-fit, and 3. time-dependent variables. 
- __Graphical techniques__: in short, the graphical method compares two graphs over time for two groups and if they are independent over time, they should show such a behavior on the plotted graph. Some of the graphical techniques are:
  - Schoenfeld Residuals Plot: The residuals should be independent of time if the PH assumption holds. A plot of Schoenfeld residuals against time should show no pattern or trend.
  - Log(-Log(Survival)) Plot: plotting the log(-log(survival)) against log(time), if the PH assumption is valid, the plot should show parallel lines for different levels of the predictor.
  - Kaplan-Meier Curves:for categorical predictors, Kaplan-Meier curves can be used. If the PH assumption holds, the survival curves for different groups should be roughly parallel.
  - Time-Dependent Covariates: Including time-dependent covariates in the model can help test the PH assumption. If the coefficients of these covariates are significant, it indicates a violation of the PH assumption.
  - Observed w/ predictor: comparing observed vs predictor survival curve.
- __Goodness-of-fit__: for a large sample of Z or chi-square statistics, based on p-values derived from standard normal statistics for each variable, if it is not significant, it indicates the PH assumption is satisfied. One downside of the GOF method could be it is too global and may not detect a specific violation of PH assumption.
- __time-dependent variable__: generating a new feature by multiplying time by a time-independent variable, creates a time-dependent feature. Now if the coefficient of this new feature is significant, the PH assumption is violated for the original feature. 
 

## Stratified Cox model
The Stratified Cox model, is a modification to Cox model that uses stratification to control predictors that do not satisfy PH assumption. Stratification could be applied to one or more variables. A stratified method could be with no-interaction, or run with interaction.  The stratified Cox model assumes that there is no interaction between the stratification variable and other covariates. This means that the effect of the stratification variable is not modified by the other covariates.

__Single-variable stratification__

In this method, if a variable is time-dependent, it is stratified in the model, while the other time-independent variables are included.
  
__General form (k-variable stratification)__

Given a dataset with _k_ variables not satisfying the PH assumption and _p_ variables that satisfy the PH assumption, we perform the stratification as follows:
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
\end{cases}$$

Employ heaviside function $E g(t)$ will be $E$ for $t \leq t_0$, and _0_ for all other cases.

__Ancillary variables__ is another type of variable for which  its value changes primarily because of “external” characteristics of the environ ment that may affect  several individuals simultaneously. An example of this type of variables is air polution.

__General formula for extended Cox PH model__
The formula for the extended Cox-PH model consists of two terms, one associated with time-dependent variables and the other time-independent variables: 
$$Extended\ Cox-PH: h(t, X) = h_0(t) e^{\sum (\beta_i X_i) + \sum (\gamma_j Z_j)}$$

Here the $X_i$ variables are time independent and $Z_j$ are time dependent.

Extended Cox-PH model formula could be written as log-time dependent variables as in the following: $Extended\ Cox-PH: h(t, X) = h_0(t) e^{\sum (\beta_i X_i) + \sum (\gamma_j Z_j (t - L_j))}$.

In any-case, the proportional hazard (PH) assumption no longer holds for the extended Cox model. 

---
# <a id = 'strcox'> [General Stratified Cox (SC) Model](#up) </a>

The general Stratified Cox (SC) model is an extension of the Cox PH model, and is used to explore the relationship between the survival time of and one or more predictor variables. The stratification part of the model handles situations where the proportional hazards assumption might not hold for some subgroups (strata) of the data. 

- Cox PH model assumes the effect of covariates on hazard is multiplicative or constant over time.
- Stratification is then used in cases in which the baseline hazard function may differ across different strata (subgroups). By using stratification, each subgroup will have its own baseline hazard function, which can improve the model’s fit when proportional hazard assumptions do not hold for all covariates. Each stratum has its own baseline hazard function. As the result, the relationship between covariates and hazard function could be different among different groups.

Original Cox PH model: $$h(t|X) = h_0(t) \exp(\beta_1 X_1 + \beta_2 X_2 + ... + \beta_p X_p)$$

Stratified Cox PH model: $$h(t|X, S) = h_{0S}(t) \exp(\beta_1 X_1 + \beta_2 X_2 + ... + \beta_p X_p)$$

Where 
- $S$ represents the stratum (subgroup) to which the observation belongs,
- $h_{0S}(t)$ is the baseline hazard for stratum $S$.



__Example__
```python

import pandas as pd
import numpy as np
from lifelines import CoxPHFitter
from lifelines.datasets import load_rossi

# Load the Rossi dataset (a common dataset for survival analysis)
data = load_rossi()
data['race'] = data['race'].astype('category')

# Instantiate the Cox Proportional Hazards model
cph = CoxPHFitter()

# Fit the model with stratification by 'race'
# duration_col: time until the event or censoring occurs
# event_col: the event of interest column
cph.fit(data, duration_col='week', event_col='arrest', strata=['race'])

# Display the summary of the model
cph.print_summary()

# Check the baseline survival function for different strata
cph.plot()

```

## No-Interaction Assumption 

The Cox PH model assumes the effect of each covariate on the hazard is constant over time, and that the effects of the covariates do not interact with each other. The no-interaction assumption says the effect of a covariate on the hazard is not influenced by other covariates. In other words, the hazard ratio for a particular covariate is assumed to be constant and independent of the values of other covariates.

__Test the No-Interaction Assumption__
- method 1 - inclusion in model: to test an interaction, we can add the interaction term to the Cox PH model, and check the results. If the p-value shows significance for that covariate, it means the no-interaction assumption is violated.
- method 2 - likelihood ratio test. If the likelihood ratio test indicates that the interaction term significantly improves the model, it suggests that the No-Interaction Assumption does not hold, and interactions should be included.
  - Null Model: A model without the interaction term.
  - Alternative Model: A model that includes the interaction term.
- method 3 - statistical test: using `CoxPHFitter` class to fit a model and add interaction terms to test for significant interactions.

__Example__

```python
import pandas as pd
import numpy as np
from lifelines import CoxPHFitter
from lifelines.datasets import load_rossi

# Load a sample dataset
data = load_rossi()

# Let's consider two covariates: 'age' and 'fin'
# First, fit the Cox model without interaction terms
cph_no_interaction = CoxPHFitter()
cph_no_interaction.fit(data, duration_col='week', event_col='arrest')

# Now, create a new feature for the interaction between 'age' and 'fin'
data['age_fin_interaction'] = data['age'] * data['fin']

# Fit the Cox model with the interaction term
cph_with_interaction = CoxPHFitter()
cph_with_interaction.fit(data, duration_col='week', event_col='arrest')

# Compare the models
print("Model without interaction:")
cph_no_interaction.print_summary()

print("\nModel with interaction:")
cph_with_interaction.print_summary()

# Conduct Likelihood Ratio Test
# We can check if the interaction term significantly improves the model by comparing AIC values or using a formal likelihood ratio test.
# If the interaction term is significant, the model with interaction will be preferred.
```


## The Stratified Cox Likelihood
The likelihood function for the Stratified Cox Model is an extension of the standard Cox likelihood, but with one key difference: it sums over the baseline hazards for different strata.

---

# <a id = 'coxextend'> [Extension of the Cox Proportional Hazards Model for Time Dependent Variables](#up) </a>











---

# <a id = 'parametric'> [Parametric survival model](#up) </a>
The parametric survival model is another survival method that is used to predict survival probabilities, hazard rates, and other quantities of interest. Unlike a non-parametric (KM model) or a semi-parametric (Cox PH model) method, the parametric could yield a more accurate estimation or prediction. This is due to the fact that the parametric approach assumes that survival time follows a specific probability distribution, such as exponential, Weibull, log-normal, or gamma distribution. This helps the model to perform better.

## Parametric model based on distribution type of survival time

A parametric model consists of two functions: $f(t) = S(t) \times h(t)$
- survival function $\( S(t) \)$, which represents the probability that an individual survives from the time origin to at least time $\( t \)$.
- hazard function $\( h(t) \)$, which represents the instantaneous risk of the event occurring at time $\( t \)$, given that the individual has survived up to time $\( t \)$.

  
Formula of some of the survival and hazard functions, based on the distribution of survival time:
- Exponential distribution: a model based on exponential distribution assumes survival time has a constant hazard rate over time.

  $$S(t) = \exp(-\lambda t)$$
  
  $$h(t) = \lambda$$
  
  where $\( \lambda \)$ is the rate parameter.
 
 - Weibull distribution: more flexible distribution that could handle increasing or decreasing hazard rate.
  
   $$S(t) = \exp(-\lambda t^\gamma)$$
 
   $$h(t) = \lambda \gamma t^{\gamma-1}$$
 
   where $\lambda$ and $\gamma$ are shape and scale parameters.
   
- Log-Normal Distribution assumes the logarithm of the survival time follows a normal distribution. The survival function does not have a closed-form expression and is often evaluated numerically.
- Gamma Distribution is a flexible model that can handle various hazard shapes. The survival function and hazard function are more complex and often require numerical methods for evaluation.


__Steps__
1. Select an appropriate probability distribution based on the characteristics of the data and the underlying process.
2. Employ maximum likelihood estimation (MLE) or Bayesian methods to estimate the parameters of the chosen distribution.
3. Evaluate the goodness of fit using diagnostic plots and statistical tests.
4. Use the fitted model to predict survival probabilities, hazard rates, and other quantities of interest.



__Example__
```python
import pandas as pd
from lifelines import WeibullFitter

# data
data = {
    'time': [5, 6, 6, 6, 7, 10, 13, 16, 20, 25],
    'event': [1, 1, 0, 1, 1, 1, 1, 1, 0, 1]
}
df = pd.DataFrame(data)

# Create/fit a WeibullFitter object and print the results
wf = WeibullFitter()
wf.fit(df['time'], event_observed=df['event'])
wf.print_summary() 
```

Output
```
WeibullFitter
==================================================================================
                coef    exp(coef)  se(coef)       z        p     -log2(p)
----------------------------------------------------------------------------------
lambda         1.6392     5.1500     0.6319    2.594     0.0095      6.6970
rho            0.7423     2.1010     0.1312    5.656    <0.0001     12.8783
----------------------------------------------------------------------------------
```

Interprete the results
- `coef` column shows the estimated parameters for the Weibull distribution.
- `exp(coef)` column shows the exponentiated coefficients, which can be interpreted as hazard ratios.
- `p` column indicates the statistical significance of each parameter. Parameters with low p-values (e.g., < 0.05) are considered statistically significant.

## Accelerated failure time assumption (AFT)



---










  























---
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







