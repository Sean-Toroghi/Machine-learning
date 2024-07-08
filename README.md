<h1>Machine Learning</h1>

<a id='up'></a>

<h4> Table of contents </h4>

 
 
1. Data science life cycle
2. Automate machine learning via FLAML
3. Supervised learning (regression - classification)
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
        2. Aglommerative clustering 
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
- define the problem: as a priliminary part of a project, this step requires a deep understanding stakeholder requirements, formulating hypothesis, and determining project scopre. The outcome will be a clear and consice buiness problem statement, goals, and objectives.
- data collection: collecting data from different sources (e.g. databases, web scraping, third party provider, ...) and ensure the data is accurate, relevant to the problem in hand, and representative. This phase requires to establish data lineage through documentation including origin of data, and how it moved around. Also, documenting data format. structure, content, meaning, and any potential bias during the sampling/collection is helpful.
- data preparation: this phase consists of data cleaning (e.g. imputation and removing duplicates), transformation (e.g. normalization and encoding categorical variables), feature engineering (e.g. aggregating or create new features), and moving/joining data as needed.
- data exploration: performing explanatory data ana;ysis (EDA) helps to gain insights into the data. This phase includes visualizing distributions, itentifying trend/patterns, detecting outliers/anomalities, and checking for potential relationship among/correlation between features.
- model selection: based on the problem in hand and characteristics of the data, we need to pick the appropriate modeling technique and choose a set of algorithms (to validate performance).
- model training: decide about spliting technique (train/val/test or cross-validation), set the parameters (hyper-parameters) and fitting model to data are done in this step.
- model evaluation: assessing model performance via appropriate technique (e.g. f1-score, AUC-ROC, RMSE) and compare them to select the best performing model/s is crusial in machine learning pipeline. To ensure an unbias evaluation, we could employ techniques such as cross-validation or hold out test sets.
- model tuning: fine-tune selected mdoel/s via optimizing hyper-parameters, feature selection, or incorporating domain knowledge are part of model tuning. 
- model deployment: Deploying the final model into a production environment is the final step of modeling stage, which will put the model in use, whetehr for making a predictio or inform decision-making. This step involves integrating the model to the exisitng sytem, creating API, or setting up monitoring and maintenance procedures.  
- model monitoring and maintenance: in many cases the data stream over time could negatively impact the performance of a model, as new data may drift away from the original data, which the model was trained/tuned on. Continous monitoring the model performance could detect model or data drift and perform mitigation strategies such as retraining the model with new data, updating features, or refining the problem definition..
- communicate results: finally as the last stage of data science lifecycle and includes share insights and results with stakeholders such as any recommendations or actions based on the analysis. Data scientist could employ techniques such as creating visualizations, dashboards, or reports to communicate the findings effectively.


---
# Automate machine learning systems  

AutoMl systems aim to simplify some of the tedious parts of ML piepline and are available at various levels of complexity, targeting one or some parts of the developing a model such as preprocessing, feature engineering, model selection, and hyperparametr fin tuning. Two examples of AutoML framework are Optuna and FLAML.

## Automating feature engineering

Data clearning and feature engineering are crusial part of ML workflow, which includes dealing with unusable data, handling missing values, and creating meaningful features. In a general form AutoML could be done either by using heuristic algorithms and tests to select he best technique, or take multiple approaches and tests which works best by training models against the data. 

Example of automated data cleaning is to perform imputation using descriptive statistic (e.g. mean or mode). 

One shortcomming of automated feature engineering is it cannot benefit from domain knwoledge and expert inputs during the process.

## Automated model selection and tuning

AutoML is very useful when it is used for model selection and hyperparameter tuning. There are several techniques for AutML in terms of model selection, incluing Bayesian optimization and meta-learning. Coupling AutoML with cross-validation technique reduces the risk of overfitting. Furthermore, some AutoML solutions also provide value in ongoing model monitoring and maintenance. AutoML can help monitor model performance and retrain the model as needed, ensuring that the deployed ML system remains effective in the long run.

Again, the downside of employ AutoML is it acts as a black box. A continious monitoring and examination is required to ensure the outcomes are align with the initial objectives. 

---
## Fask and lightweight AutoML (FLAML)

FLAML is an AutoML solution proposed by Microsoft. The primary aim of FLAML is to minimize the resources required to tune hyperparameters and identify optimal ML models, making AutoML more accessible and cost-effective, particularly for users with budget constraints.
