<h1>Recommender system - summary</h1>

__References__
- [Book: Machine Learning: Make Your Own Recommender System - 2024](https://learning.oreilly.com/library/view/machine-learning-make/9781835882061/)
- []()
- []()
- []()
- []()

# Recommender system - intro

Recommender systems, unlike other machine learning methods such as decision trees, are a mismatch of algorithms that are all act under a common goal: to make a relevant recommendation. At a high level, recommenders sytems can be divided into two main categories:
1. collaborate filtering: recommend items to an individual based on the items similar users with shared interests, purchased on consumed.

   Advantages
   - since it does not rely on understanding items and their attributes, it is usefull in the case we have low knowledge about item characteristics
   - it is a flexible method that can handle and adopt to changes over time.
   - it generates recommendation of items outside the user's standard periphery
  
   Disadvantages
   - it requires enough data to build a user preference dataset
   - it is highly volunerable to malicious activity. To mitigate the sensitivity of model against shillinh attach is to limit the model's analysis to users purchase, not the browsing habit or other activities that can be fabricated easily.
   - in some cases it lacks consistency, as users have different standards.For example, a three-star rating could have different meaning for different people. 
     
3. content-based filtering: recommend similar items to an inddividual based on items the user has already purchased or consumed.

There is a trade-off between the two approaches, as the collaborate filtering method requires information about the users up-front. Content-based filtering requires information about new items upfront. 

__Other methods__

Based on the main two approaches mentioned above, a rane of other methods are introduced over time:
- hybrid approach
- Knowledge-based Recommenders
- Demographic Recommenders
- Mobile-based Recommenders

- location-based recommenders

- Time-sensitive recommenders

  Recommend items based on the 
  
- constraint-limited recommenders

  In this approach, the recommendation is bounded to one or some constraints, such as bounding to user's budget or bounding to specific date range.
  
- group-based recommenders

  This method aggregates individual preferences to recommend content/activities to a group of users, such as advertisement plays on screen at a shopping mall.
  
- social recommenders

  Making recommendation based on social structures and mutual relationships, such as suggest stories on X or facebook based on mutal friends.


# Data

## Collecting data
In a basic format, the data requires to have a list of users and items and the corresponding rating/feedbacl for each pair. However, the dataset in most cases is sparse with many Nan (not all users provide rating/feedback for all items). To tackle the issue of sparse dataset, we can generate inference about user preferences by using indirect feedback (__latent variables__). One example of latent feedback is using past purchases as an indicator of user preference, or associate watched videos or time spend on a page with preference. Another approach to gain preference is text mining and semantic analysis. In this approach the content of a text-based comment by user, pretaining to a video or product is used to make inference about his/her preference.

One challenge of computing preference via indirect feedback is the interpretation of continuous variable to boolean or ordinal categorical variable. As an example, converting the time spend on a video to a feedback with 5 classes could contain interpretation feedback.  feedback system

## data structure

Input data comes in two major categories: structured and unstructred. __Structured data__ is information that resides in a fixed field within a record or file. This format has a predefine schema. __Unstructured data__ or non-structured data is information that doesn’t fit neatly into a pre-defined data model or isn’t organized in a pre-defined manner. An example of unstructured data is information in email or social media posts. 

## data reduction

Data reduction helps to separate the signal from the noise, reduce total processing time, and minimize the consumption of computational resources. This is a crusial step, specifically in the case the data originates from a popular e-commerce platforms or social media networks. Another use of data reduction is to aid better visualization. 

One downside of data reduction is its potential negative effect on accuracy and relevancy.

Some of the data reduction techniques are
- row compression
- dimension reduction
- principle component analysis

__Row compression__ 

Row compresion involves reducing the volume of rows while attempting to preserve information from the original structure. One approach is to employ clustering, and based on the results perform an aggregation method.

__Dimension reduction__

Dimension reduction (also called descending dimension algorithms) transforms data from high-dimensional to low-dimensional. A range of approaches are available to reduce dimension, from manual manipulation to employ an algorithm. An exmple is to group together some features as a single feature.

__Principle component analysis (PCA)__

PCA (also caleld general factor analysis) examines interrelations among a set of variables and removes components thtat have the least impact on data variablit. The befit of PCA is it helps to reveal hidden and simplified structures in the data and is often used as a pre-processing step before applying another algorithm. t also helps to reduce data complexity. Furthermore, PCA can be used to visualize a high dimension dataset.  

# Item-based collaborative filtering
The item-based collaborate filtering generates recommendations based on the similar item to the one a user has already purchased/selected (translated to user preference). This method __first takes a given item__, then finds users who liked that item. In the final step, it retrives other items that those users liked. 

Item-base and user-based collaborate filtering both generate similar item recommendations, while the item-based method is more suited for dataset with less information regarding user charactristics and tastes.

# User-based collaborative filtering
User-based filtering __first takes a selected user__, finds users similar to that user based on similar ratings, and then recommends items that similar users also liked.

User-based filtering is more accurate than item-based filtering, when the dataset contains a large number of users with esoteric interests. 


# Content-based filtering

