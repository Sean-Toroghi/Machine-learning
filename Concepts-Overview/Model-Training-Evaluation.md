<h1>Model training and evaluation</h1>


Machine learning _lifecycle_, in a general form, consists of the following four phases:
1. data collection, preprocessing, cleaning, and processing
2. algorithm selection according to the machine learning task in hand
3. training and optimization of the model, which consists of several steps:
   - hyper parameter tuning
   - experiment  tracking
4. Model evaluation and determine the best performing model, its efficiency on a new dataset and in production.

This section covers topics related to the last two phases: __model training and evaluation__.


---
## Define machine learning task
Before starting to develop a machine learning solution, the  first question to answer needs to be "why machine learning?". There are many cases where the heuristic or statistical (ruled-based) methods perform as good, if not better, than a complex machine learning algorithm. Furthermore, if machine learning route is justified, the starting point needs to be a simple algorithm. This allows to first have a baseline to compare the different algorithm against it, and also evaluate end examine if the performance gain from more complex methods worth the effort of implementing and minting them.

## Data preprocessing and processing (feature engineering)
Without data we do not have a machine learning model. After establishing the data acquisition, we need to perform __Explanatory data analysis (EDA)__ with goal of evaluating if the existing data is sufficient as the starting point.

__EDA__

The goal of explanatory data analysis is to get a high level overview of the distribution of data and also find any flatus such as missing values rates, skewness of the distribution, outliers, Duplications, consistency, and so on. Also it helps to familiarize with dataset and features, and also be aware of any potential issues that could negatively effect the model and its performance.

__Data processing (Feature engineering)__

Machine learning methods may each require a specific type of input. To ensure compatibility of data with machine learning models, we may need to make some changes to the dataset. Also, we need to address flaws in the dataset (such as missing value imputation, removing duplicates, ). Furthermore, we may generate new features, based in the knowledge we gain during the EDA phase. Selecting the most appropriate features (employing matrix-correlation, investigating coefficient, or using the feature importance function in the ensemble tree algorithm) helps to improve model performance by avoiding the curse of dimensionality. Finally,  dimensionality reduction may be required to limit the number of features in the model and avoid redundancy. 

__Preprocessing dataset__

Transforming the data to the form requires a machine learning model, such as encoding categorical variables, is done during the preprocessing phase. Another important task  is standardization of the features, which in some machine learning methods is a requirement. tragi encoding/mapping also may be needed. 

---
# Model training
After processing the data and have the data ready for machine training, we need to train the model. However, this phase also consists of several phases, including:
- define machine learning task
- select most suitable machine learning model
- train the model

## Define machine learning task

## Model selection

## Model training 





--- 
# Iterative characteristic of training 

Training a machine learning model, may leads us back to data acquisition. In many cases, after training a model, I find out the size of distribution of the training dataset is not adequate for a machine learning algorithm to learn and be able to generalize during the inference. In such a case, collecting new data or generating synthetic dataset helps the model to perform better.

---

# Model evaluation


