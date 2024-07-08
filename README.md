<h1>Machine Learning</h1>

<a id='up'></a>

<h4> Table of contents </h4>

 
 
1. Data science life cycle
2. Machine learning operations (MLOps)
3. Automate machine learning via FLAML
4. Supervised learning (regression - classification)
    1. Regression (Linear, Ridge/LASSO, Polynomial, SVM, K-NN)
    2. Classification (K-NN, Logistic regression, SVM, Kernelized SVM, Decision Trees, Naive Bayes Classifiers, 
    3. Ensemble models
        1. XGBoost
        2. LightGBM
        3. Random forest
5. Un-supervised learning
     1. Transformation
        1. Dimensionality Reduction: PCA
        2. Manifold learning: MDS and t-NSE
     2. Clustering
        1. K-means clustering
        2. Agglomerative clustering 
        3. DBSCAN
        4. K-mode clustering
        5. K-prototype clustering
        6. Evaluate clustering algorithm

---

# Data science life cycle

At the heart of data science lies the data science life cycle, a systematic, iterative process that guides data-driven problem-solving across various industries and domains.

1. The first stage of the data science life cycle involves defining the problem, which entails understanding the business context, articulating objectives, and formulating hypotheses.
2. Once the data is analyzed, the data science life cycle progresses to model selection, training, evaluation, and tuning.
3. Lastly, the data science life cycle emphasizes the importance of deploying the final model into a production environment, monitoring its performance, and maintaining it to ensure its ongoing relevance and accuracy.

![image](https://github.com/Sean-Toroghi/Machine-learning/assets/50586266/e466f95a-5276-43b6-9a0c-73a3b62fb118)
[Ref.](https://learning.oreilly.com/library/view/machine-learning-with/9781800564749)


Key steps in the data science lifecycle:
- define the problem: as a preliminary part of a project, this step requires a deep understanding of stakeholder requirement, formulating hypothesis, and determining project score. The outcome will be a clear and concise business problem statement, goals, and objectives.
- data collection: collecting data from different sources (e.g. databases, web scraping, third party provider, ...) and ensure the data is accurate, relevant to the problem in hand, and representative. This phase requires to establish data lineage through documentation including origin of data, and how it moved around. Also, documenting data format. structure, content, meaning, and any potential bias during the sampling/collection is helpful.
- data preparation: this phase consists of data cleaning (e.g. imputation and removing duplicates), transformation (e.g. normalization and encoding categorical variables), feature engineering (e.g. aggregating or create new features), and moving/joining data as needed.
- data exploration: performing explanatory data analysis (EDA) helps to gain insights into the data. This phase includes visualizing distributions, identifying trend/patterns, detecting outliers/anomalies, and checking for potential relationship among/correlation between features.
- model selection: based on the problem in hand and characteristics of the data, we need to pick the appropriate modeling technique and choose a set of algorithms (to validate performance).
- model training: decide about splitting technique (train/val/test or cross-validation), set the parameters (hyper-parameters) and fitting model to data are done in this step.
- model evaluation: assessing model performance via appropriate technique (e.g. f1-score, AUC-ROC, RMSE) and compare them to select the best performing model/s is crucial in machine learning pipeline. To ensure an unbais evaluation, we could employ techniques such as cross-validation or hold out test sets.
- model tuning: fine-tune selected mdoel/s via optimizing hyper-parameters, feature selection, or incorporating domain knowledge are part of model tuning. 
- model deployment: Deploying the final model into a production environment is the final step of modeling stage, which will put the model in use, whether for making a prediction or inform decision-making. This step involves integrating the model to the existing system, creating API, or setting up monitoring and maintenance procedures.  
- model monitoring and maintenance: in many cases the data stream over time could negatively impact the performance of a model, as new data may drift away from the original data, which the model was trained/tuned on. Continuous monitoring the model performance could detect model or data drift and perform mitigation strategies such as retraining the model with new data, updating features, or refining the problem definition..
- communicate results: finally as the last stage of data science lifecycle and includes share insights and results with stakeholders such as any recommendations or actions based on the analysis. Data scientist could employ techniques such as creating visualizations, dashboards, or reports to communicate the findings effectively.


---
# Automate machine learning systems  

AutoMl systems aim to simplify some of the tedious parts of ML pipeline and are available at various levels of complexity, targeting one or some parts of the developing a model such as preprocessing, feature engineering, model selection, and hyperparameter fin tuning. Two examples of AutoML framework are Optuna and FLAML.

## Automating feature engineering

Data clearning and feature engineering are crucial part of ML workflow, which includes dealing with unusable data, handling missing values, and creating meaningful features. In a general form AutoML could be done either by using heuristic algorithms and tests to select he best technique, or take multiple approaches and tests which works best by training models against the data. 

Example of automated data cleaning is to perform imputation using descriptive statistic (e.g. mean or mode). 

One shortcoming of automated feature engineering is it cannot benefit from domain knowledge and expert inputs during the process.

## Automated model selection and tuning

AutoML is very useful when it is used for model selection and hyperparameter tuning. There are several techniques for AutML in terms of model selection, including Bayesian optimization and meta-learning. Coupling AutoML with cross-validation technique reduces the risk of overfitting. Furthermore, some AutoML solutions also provide value in ongoing model monitoring and maintenance. AutoML can help monitor model performance and retrain the model as needed, ensuring that the deployed ML system remains effective in the long run.

Again, the downside of employ AutoML is it acts as a black box. A continuous monitoring and examination is required to ensure the outcomes are align with the initial objectives. 

---
## Fast and lightweight AutoML (FLAML)

FLAML is an AutoML solution proposed by Microsoft. The primary aim of FLAML is to minimize the resources required to tune hyperparameters and identify optimal ML models, making AutoML more accessible and cost-effective, particularly for users with budget constraints. It automatically choose the best algorithm for a given dataset and optimize its hyperparameters, providing users with an optimal model without extensive manual intervention.

Key feature of FLMAL are:
- efficiency: the efficiency of FLMAL is by large comes from its cost-effective search algorithms. 
- versatility: FLMAL works with different environment, and support a range of ML algorithms including CatBoost, RandomForest, XGBoost, LightGBM, and several linear models.

__Hyperparameter optimization algorithm in FLMAL__
- Cost frugal optimization

  CFO is a local search method that leverages random direct search to explore the hyperparameter space. It starts at a low cost configuration and walks toward higher cost options via randomly taking steps for a fixed number of iterations among available options. The step-size is adaptive. CFO also employs random restarts to avoid stocking in local optimal.
- BlendSearch

  BlendSearch is an alternative to the CFO approach, that runs both local and global search. Similar to CFO it starts with low cost option. However, it does not wait for the local search to stagnate before exploring new regions. Instead, a global search algorithm (such as Bayesian optimization) continually suggests new starting points. These starting points are filtered based on their proximity to the current standing and prioritized by their cost. Each iteration of BlendSearch then chooses whether to continue a local search or start at a new global search point based on the performance in the previous iteration. 

Comparison:
- BlendSearch is recommended over CFO if the hyperparameter search space is highly complex.
- CFO is faster and more efficient. A good practice is to start with CFO, nd only switch to BlendSearch if CFO is struggling.
  
__FLAML limitations__
- For complex, domain-specific tasks, manual tuning may deliver better results.
- Its output is not as interpretable as other methods such as Optuna.
- FLAML only performs the model selection and tuning part of the ML process.


---

## Featuretools



Featuretools is an AutoML framework designed for automated feature engineering, and very practical in the case of transforming relational datasets and temporal data. 

- [Ref. 2](https://featuretools.alteryx.com/en/stable/#)
- [Ref.](https://featuretools.alteryx.com/en/stable/api_reference.html)

  
---
# Machine Learning operations (MLOps)

Concept of MLOps refers to practical actions aims to blend the filed of machine learning with system operation. The __Goal__ is to standardize and streamline the lifecycle of ML model development as well as it deployment. The result will be increase the efficiency and effectiveness of ML solutions with the business setting. 

The focus of MLOps is model creation, experimentation, and evaluation along with deployment, monitoring, and maintenance. This particularly important for ML solutions, since ML systems are more dynamic and less predictable than traditional software systems. 

In nutshell MLOps aiming to accelerate the ML life cycle, facilitating faster experimentation and deployment, through 
- automation at different stages,
- ensuring reproducibility,
- emphasizing on versioning code, data, and model configurations,
- monitoring its performance and continuously validating its predictions,
- use of robust testing practices for ML (e.g. containerization and serverless computing platforms),

In summary, MLOps provides a framework for managing the end-to-end ML life cycle, from initial experimentation to robust, scalable deployment. MLOps emphasizes standardization, automation, reproducibility, monitoring, testing, and collaboration to enable high-throughput ML systems.




---
__References__
- C. Wang, Q. Wu, M. Weimer, and E. Zhu, “FLAML: A Fast and Lightweight AutoML Library,” in MLSys, 2021.
- Q. Wu, C. Wang and S. Huang, Frugal Optimization for Cost-related Hyperparameters, 2020.
- C. Wang, Q. Wu, S. Huang, and A. Saied, “Economical Hyperparameter Optimization With Blended Search Strategy,” in ICLR, 2021.
