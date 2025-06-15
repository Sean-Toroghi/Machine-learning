# Spark

## Overview

__Machine learning workflow and pipeline__: The _machine learning workflow_ comprises a set of stages that help us reach the goal of having a machine learning model running in production solving a business problem. The automation of this workflow is referred to as the _machine learning pipeline_. To improve the accuracy of the model, the workflow is iterative. The workflow consists of several stages:
- collect and load/ingest data
- explore and validate data
- extract features and peform feature engineering
- split data into train/validation/test sets
- train and tune model via train and validation sets
- evaluate model via test set
- deploy model
- monitor model

__Distributed computing__ is the use of distributed systems, where multiple machines work together as a single unit, to solve a computational problem. A program that runs inside such a system is called a distributed program, and the process of writing such a program is known as distributed programming. Two general categories of distributed computing models are:
- __General-purpose distributed computing models__ allow users to write a custom data processing flow using a defined abstraction. Examples:
  - MapReduce: introduced by Google in 2004
  - Message Passing Interface (MPI) programming model.
  - Barrier
  - Shared memory models
- __Dedicated distributed computing models__ are models that were developed to support a specific need in the machine learning development cycle. Examples
  - parameter server in Tensorflow

__Distributed system architecture__

  
__Goal__: find the best way to divide a problem into separate tasks that multiple machines can solve in parallel through message communication. 
