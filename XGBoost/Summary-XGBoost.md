<h1>XGBoost</h1>

References
- 

# Overview
XGBoost is an extention of gradient boosted cdecision trees, with the goal of improving its speed and performancce. 

__Model evaluation_
- Generally k-fold cross-validation is the gold-standard for evaluating the performance of a machine learning algorithm on unseen data with k set to 3, 5, or 10.
- Use stratified cross-validation to enforce class distributions when there are a large number of classes or an imbalance in instances for each class.
- Using a train/test split is good for speed when using a slow algorithm and produces performance estimates with lower bias when using large datasets.

---
- __save model__:
  - regular save: `pickle.dump(model, open("model_checkpoint.dat", "wb"))`
  - seriealized save with joblib: `joblib.dump(model, "model_checkpoint.dat")`
- __load model__:
  - reular load: `loaded_model = pickle.load(open("model_checkpoint.dat", "rb"))`
  - load serialized save model: `loaded_model = joblib.load("model_checkpoint.dat")`
- 
