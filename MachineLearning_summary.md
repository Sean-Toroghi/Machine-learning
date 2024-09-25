<h1>Machine learning - quick reference</h1>

# Steps involve in a machine learning project

A machine learning prject, in its general form, goes through the following phases:
0. define problem
1. collecting and aquiring data
2. preprocess and clean data
3. select algorithm apprpriate for the task in hand, and process of selecting best algorithm among all cadidates
  - develop a baseline to evalue model performance in the next step (either a simple algorithm with low cost and/or heuristic-based (i.e., rules-based) approach)
  - training process and optimizing the model to gain highest performance, including fine-tuneining, hyperparameter tuning, experiment tracking. 
4. evaluate model and examine its generalization capability

# Step 0 - define problem
Perhaps define and translate a problem into ML problem is the most critical point in any ML project. It requires to clearly define problem, scope, budget, requirement, cost, potential for future expantion, and feasibility. Sometime, a ruled-based model can deliver the same results as of a complex and sophisticated model generate. Also, have a clear understanding of problem requires, in many case, gather domain knowledge. This later will help when performing feature engineering and also choosing ML algorithms. Finally, define boundaries (scopre, budget, cost, time, ...) influence the choice of approach throughout the project, from data aquisition to model update.

# Step 1 - collecting and acquiring data

__References__
- [Making Scence of Data - By Glenn J. Myatt, Wayne P. Johnson](https://learning.oreilly.com/library/view/making-sense-of/9781118422106/)
- [Hands-On Data Preprocessing in Python - By Roy Jafari]([https://learning.oreilly.com/library/view/hands-on-data-preprocessing](https://learning.oreilly.com/library/view/hands-on-data-preprocessing/9781801072137/))
- [Hands-On Exploratory Data Analysis with Python - By Suresh Kumar Mukhiya, Usman Ahmed](https://learning.oreilly.com/library/view/hands-on-exploratory-data/9781789537253/)

There are many options available for data acquisition in the context of ML, such as access to proprietary data, public datasets, web scraping, purchasing data from vendors, and creating synthetic data.

# Step 2 - data preprocessing and feature engineering

## EDA
Explanatory data analysis is the first step in data processing. In a general form, it consists of 
- gain a high-level overview of distribution of data: mean, s.d., max, min, ... for continous, and count, number of unique categories, ... for categorical variables
- find any potential flaws, such as missing vlaues, skewness, duplicates, and outliers
To interpret the resutls of EDA, it is important to have some domain knwoledge. Simply looking at statistical results may not be enought

__Tools__
- Pandas and Polars functions for EDA
- [ydata-profiling]()
- Create custom function to generate a report
- Visualization (python packages (Matplotlib, Seaborn, Plotly), Softwares (Tableau))
  


#
