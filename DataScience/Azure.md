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
 - Azur Machine Learning: the pipeline can be crated with the Designer, or by creating a collection of scripts. Pipeline can be used for ETL, as well as other task in this service.

Azur Synapse and Databrick provide more scalable compute than Machine-Learning serivce. 


Eample: while the services can peform diferent tasks, each one has a targeted task that their design makes them suitable foor that specific task. For example a _data ingestion solution_ could be as follow:
- extract raw data
- copy/transform the data with _Synapse Analytic_
- store the data in _Blob storage_
- train the model in _Machine Learning_

__Data storage options__
- Blob Storage
 - Definition: Blob (Binary Large Object) storage is designed to store large amounts of unstructured data, such as images, videos, audio files, and backups.

 - Use Cases: Ideal for serving images or documents directly to browsers, storing files for distributed access, and handling big data like logs and database backups.

 - Structure: Data is organized into containers (similar to folders), and each blob is associated with a unique URL.

- File Storage
 - Definition: File storage is designed to store data in a hierarchical file system, similar to how files are stored on a traditional computer.
 - Use Cases: Suitable for applications that require a shared file system, such as virtual machines or applications that need to access files in a traditional file structure.
 - Structure: Data is organized in directories and subdirectories, with files having paths.

- Data Lake
 - Definition: A data lake is a centralized repository that allows you to store all your data (structured and unstructured) at any scale.
 - Use Cases: Ideal for big data analytics, data science, and machine learning, where you need to store and analyze vast amounts of data.
 - Structure: Data lakes typically use a flat namespace, meaning all data is stored in a single location without a predefined structure.

- SQL Database
 - Definition: SQL (Structured Query Language) databases are designed to store and manage structured data using a relational model.
 - Use Cases: Suitable for applications that require complex queries, transactions, and relationships between data, such as customer relationship management (CRM) systems, inventory management, and financial applications.
 - Structure: Data is organized into tables with rows and columns, and relationships are defined between tables.

