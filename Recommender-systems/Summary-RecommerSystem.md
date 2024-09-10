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
2. content-based (item-based) filtering: recommend similar items to an inddividual based on items the user has already purchased or consumed.

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

Input data comes in two major categories: structured and unstructred. Structured data is information that resides in a fixed field within a record or file. 













