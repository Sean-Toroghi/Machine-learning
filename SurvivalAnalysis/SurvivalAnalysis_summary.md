

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
The Cox PH model provides expression for hazard at point of time _t_ with a given specification of a set of explanatory variables. According to the Cox model, hazard at time _t_ is a function of two parameters: baseline hazard, and the exponential _e_ to the linear form of independent variables $\beta_i X_i$ power of 

# Evaluate proportional hazard

Among different methods for evaluating proportional hazard (PH), three famous ones are: 1. graphical,2. goodness-of-fit, and 3. time-dependent variables.


































