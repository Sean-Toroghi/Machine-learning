<h1>Machine learning - better approach</h1>

This summary explores machine learning through the unifying framework of the energy model. This perspective effectively connects diverse machine learning concepts. Unlike conventional approaches that typically begin with supervised learning, introduce various model architectures, and conclude with unsupervised learning, this framework establishes the energy model as a foundational concept for all machine learning methodologies. By adopting this approach, it becomes evident how disparate machine learning paradigms—including supervised learning, unsupervised learning, classification, regression, generative models, probabilistic models, and specialized areas like reinforcement learning—all share the fundamental core of the energy model.

## <a name="table">Table of contents</a>
- [The concept of _energy model_ in machine learning](#energy)
- Machine learning methods - foundation
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

There are three aspects to evert machine learning problems:
1. define energy function: parametarization
2. estimating $\theta$ form data: learning
3. infer missing part of the data, given a partial observation: inference


Given the energy function, we can define following concepts in machine learning:
- classification an regression by partitioning data $(x,y)$ (no latent variable $Z$) with given $y in V$ (discrete $V$ results in classification, and continuous $V$ results in regression): $\hat{y} = argmin e((x,y),\theta)$
- clustering, in case of finite set of discrete latent variable $Z$: $\hat{z} = argmin e(x,z,\theta)$
- representation learning, in the case of finite set of continuous latent variable $Z$: $\hat{z} = argmin e(x,z,\theta)$
In all three cases, the objective is to minimize the energy function, given a subset of inputs.

__Concept of learning__

In machine learning, the concept of learning means finding bet hyper-parameters $\theta$ subjct to the objective. 

__Regularization__

When performing the learning steps, to make sure the energy assign to undesireable obeservation stays relatively high, we employ _regularization_ $R(\theta)$:
$$min E_x [e(x,\theta) - R(\theta)]$$
The regularization term depends on the problem in hand.



---
# Machine learning methods - foundation

## <a name="classification">Classification</a>

## <a name="back">Backpropogation</a>
---
# Probabilistic machine learning
In probabilistic approach for developing a machine learnin model, the output is a distribution over possible outcomes instead of discrete values. One benefit of this approach is it allows to express confidence when generating the output. in Energy-based aproach, the probability of a configuration is defined in terms of an energy function: $P(y \mid x) = \frac{\exp(-E(x, y))}{Z(x)}$, where $Z$ (partition function) ensures the probabilities sum to 1: $Z(x) = \sum_y \exp(-E(x, y))$.

A probabilistic model has two constraints: 
1. non negativity: $P_\theta \leq 0$
2. normalization: $\sum P_\theta(y'\x) = 1$

__Example__
```python
import numpy as np
import matplotlib.pyplot as plt

# Suppose these are input features (e.g. after passing through a neural net)
x = np.array([0.3, -1.2])

# Define an energy function for each class y given input x
def energy(x, y):
    weights = [
        np.array([1.0, -1.0]),  # weights for class 0
        np.array([-0.5, 0.5]),  # class 1
        np.array([0.2, 0.8])    # class 2
    ]
    bias = [0.1, -0.2, 0.3]  # biases for each class
    return -np.dot(weights[y], x) - bias[y]  # negative sign because lower energy = higher prob

# Compute unnormalized probabilities
energies = np.array([energy(x, y) for y in range(3)])
unnormalized_probs = np.exp(-energies)

# Normalize
probs = unnormalized_probs / np.sum(unnormalized_probs)

# Resulting Categorical distribution
print("Class probabilities:", probs)
```

## <a name="prob">Energy model for probabilistic machine learning models</a>
### Define probabilistic model
A simple probabilistic approach for machine learning model based on energy model is achieved by transforming the energy model into a probabilistic one. As an example, for a classficiation problem we generate categorical distribution over $\Theta$ given $X$: $P_\theta (y|x)$

To define a more complex probabilistic model, seveal approaches have been developed over the past years, including:
- computing joint distribution
- computing

### Variational inference

### Gaussian mixture models

### Continuous latent variable models

### Importance sampling 

### Vaiational auto-encoder




