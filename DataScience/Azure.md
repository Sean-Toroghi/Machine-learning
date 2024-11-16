<h1>Data science: design machine learning solution</h1>

# Overview
Machine learning process:
- Define the problem: Decide on what the model should predict and when it's successful.
- Get the data: Find data sources and get access.
- Prepare the data: Explore the data. Clean and transform the data based on the model's requirements.
- Train the model: Choose an algorithm and hyperparameter values based on trial and error.
- Integrate the model: Deploy the model to an endpoint to generate predictions.
- Monitor the model: Track the model's performance.

__Storing data__
Azur provides options for storing data, among which are:
- Azure Blob Storage: Cheapest option for storing data as unstructured data. Ideal for storing files like images, text, and JSON. Often also used to store data as CSV files, as data scientists prefer working with CSV files.
- Azure Data Lake Storage (Gen 2): A more advanced version of the Azure Blob Storage. Also stores files like CSV files and images as unstructured data. A data lake also implements a hierarchical namespace, which means it’s easier to give someone access to a specific file or folder. Storage capacity is virtually limitless so ideal for storing large data.
- Azure SQL Database: Stores data as structured data. Data is read as a table and schema is defined when a table in the database is created. Ideal for data that doesn’t change over time.



__Transform data: data ingestion pipeline__

A data ingestion pipeline is a sequence of tasks that move and transform the data. To use the data ingestion pipeline, first we need to pick Azur serivce
 - Azure Synapse Analytics: create pipeline via GUI. For transforming data, this tool has UI tool to choose from. 
 - Azur Databrick: this service provides option to perform a task via coding (SQL, Python, or R). This service employs Spark cluster for distribution computation.
 - Azur Machine Learning: this service 
