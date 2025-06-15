<h1>Machine learning - unconentional approach</h1>

This summary explores machine learning through the unifying framework of the energy model. This perspective effectively connects diverse machine learning concepts. Unlike conventional approaches that typically begin with supervised learning, introduce various model architectures, and conclude with unsupervised learning, this framework establishes the energy model as a foundational concept for all machine learning methodologies. By adopting this approach, it becomes evident how disparate machine learning paradigms—including supervised learning, unsupervised learning, classification, regression, generative models, probabilistic models, and specialized areas like reinforcement learning—all share the fundamental core of the energy model.

## <a name="table">Table of contents</a>
- [The concept of _energy model_ in machine learning](#energy)
- Machine learning - foundation
  - [Classification](#classifiation)
  - [Backpropogation](#back)
  - [Stochastic gradient descent](#sgd)
  - [Generalization](#generalization)
  - [Model selecton and hyper-parameter tuning](#hp)
- Probabilistic machine learning
  - [Energy model for probabilistic machine learning models](#prob)
  - [Generative models](#gen)
  - [Continuous latent variational model](#clvm)
- [Unirected generative models](#ugen)
- Other topics
  - [Ensemble mdoels](#ensemble)
 

--- 
# <a name="energy">The concept of _energy model_ in machine learning</a>

__Energy model__ or _negativ compatibility score_ assigns a score to function $e$ that maps pair of _(oberve, latent)_ variables, parameterised by $\theta$, to a real value: $e: ((X, Z), \theta) \rightarrow \Re$. We do not observe the latent variable ($Z$) directly. The mean and variance for the eebergy funtion are defined as following:
- $\mu_e = \mathbb{E} [e(X,Z,\theta)] = \sum p(z)e(x,z,\theta)$
- $Var(e) = E [(e - \mu_e)^2]$

Given the energy function, we can define differnet concepts in machine learning:
- classification an regression by partitioning data $(x,y)$ (no latent variable $Z$) with given $y in V$ (discrete $V$ results in classification, and continuous $V$ results in regression).
- clustering, in case of finite set of discrete latent variable $Z$: $\hat{y} = argmin e(x,q,\theta)$
- representation learning, in the case of finite set of continuous latent variable $Z$

---
# Machine learning - foundation

## <a name="classification">Classification</a>

## <a name="back">Backpropogation</a>


