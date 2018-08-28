# HyperLearn
---

Faster, Leaner Scikit Learn (Sklearn) morphed with Statsmodels & Deep Learning drop in substitute. Designed for big data, HyperLearn can use 50%+ less memory, and runs 50%+ faster on some modules. Will have GPU support, and all modules are parallelized.

HyperLearn is written completely in PyTorch, NoGil Numba, Numpy, Pandas, Scipy & LAPACK, and mirrors (mostly) Scikit Learn.
HyperLearn also has statistical inference measures embedded, and can be called just like Scikit Learn's syntax (model.confidence_interval_)

![alt text](https://pytorch.org/static/img/logos/pytorch-logo-dark.png "Logo Title Text 1")

| Algorithm         |  n    |  p  | Time(s) |            | RAM(mb) |            | Notes                   |
| ----------------- | ----- | --- | ------- | ---------- | ------- | ---------- | ----------------------- |
|                   |       |     | Sklearn | Hyperlearn | Sklearn | Hyperlearn |                         |
| QDA (Quad Dis A)  |1000000| 100 |   54.2  | *22.25*    |   2,700 |  *1,200*   | Now parallelized        |

Time(s) is Fit + Predict. RAM(mb) = max( RAM(Fit), RAM(Predict) )
  
---
#### Help is really needed! Email me or message me @ danielhanchen@gmail.com!

---
# Key Methodologies and Aims
#### 1. [Embarrassingly Parallel For Loops](#1)
#### 2. [50%+ Faster, 50%+ Leaner](#2)
#### 3. [Why is Statsmodels sometimes unbearably slow?](#3)
#### 4. [Deep Learning Drop In Modules with PyTorch](#4)
#### 5. [20%+ Less Code, Cleaner Clearer Code](#5)
#### 6. [Accessing Old and Exciting New Algorithms](#6)
---
<a id='1'></a>
### 1. Embarrassingly Parallel For Loops
  * Including Memory Sharing, Memory Management
  * CUDA Parallelism through PyTorch & Numba
  
<a id='2'></a>
### 2. 50%+ Faster, 50%+ Leaner  
  * Matrix Multiplication Ordering: https://en.wikipedia.org/wiki/Matrix_chain_multiplication
  * Element Wise Matrix Multiplication reducing complexity to O(n^2) from O(n^3): https://en.wikipedia.org/wiki/Hadamard_product_(matrices)
  * Reducing Matrix Operations to Einstein Notation: https://en.wikipedia.org/wiki/Einstein_notation
  * Evaluating one-time Matrix Operations in succession to reduce RAM overhead.
  * If p>>n, maybe decomposing X.T is better than X.
  * Applying QR Decomposition then SVD might be faster in some cases.
  * Utilise the structure of the matrix to compute faster inverse (eg triangular matrices, Hermitian matrices).
  * Computing SVD(X) then getting pinv(X) is sometimes faster than pure pinv(X)
  
<a id='3'></a>
### 3. Why is Statsmodels sometimes unbearably slow?
  * Confidence, Prediction Intervals, Hypothesis Tests & Goodness of Fit tests for linear models are optimized.
  * Using Einstein Notation & Hadamard Products where possible.
  * Computing only what is neccessary to compute (Diagonal of matrix and not entire matrix).
  * Fixing the flaws of Statsmodels on notation, speed, memory issues and storage of variables.

<a id='4'></a>
### 4. Deep Learning Drop In Modules with PyTorch
  * Using PyTorch to create Scikit-Learn like drop in replacements.

<a id='5'></a>
### 5. 20%+ Less Code, Cleaner Clearer Code
  * Using Decorators & Functions where possible.
  * Intuitive Middle Level Function names like (isTensor, isIterable).
  * Handles Parallelism easily through hyperlearn.multiprocessing

<a id='6'></a>
### 6. Accessing Old and Exciting New Algorithms
  * Matrix Completion algorithms - Non Negative Least Squares, NNMF
  * Batch Similarity Latent Dirichelt Allocation (BS-LDA)
  * Correlation Regression
  * Feasible Generalized Least Squares FGLS
  * Outlier Tolerant Regression
  * Multidimensional Spline Regression
  * Generalized MICE (any model drop in replacement)
  * Using Uber's Pyro for Bayesian Deep Learning

---
# Goals & Development Schedule

### Will Focus on & why:

#### 1. Singular Value Decomposition & QR Decomposition
    * SVD/QR is the backbone for many algorithms including:
        * Linear & Ridge Regression (Regression)
        * Statistical Inference for Regression methods (Inference)
        * Principal Component Analysis (Dimensionality Reduction)
        * Linear & Quadratic Discriminant Analysis (Classification & Dimensionality Reduction)
        * Pseudoinverse, Truncated SVD (Linear Algebra)
        * Latent Semantic Indexing LSI (NLP)
        * (new methods) Correlation Regression, FGLS, Outlier Tolerant Regression, Generalized MICE, Splines (Regression)
---
1. Port all important Numpy functions to faster alternatives (ONGOING)
  * Singular Value Decomposition (50% Complete)
    *
2. 

