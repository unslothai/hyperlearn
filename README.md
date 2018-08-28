# HyperLearn

Faster, Leaner Scikit Learn (Sklearn) morphed with Statsmodels & Deep Learning drop in substitute. Designed for big data, HyperLearn can use 50%+ less memory, and runs 50%+ faster on some modules. Will have GPU support, and all modules are parallelized.

HyperLearn is written completely in PyTorch, NoGil Numba, Numpy, Pandas, Scipy & LAPACK, and mirrors (mostly) Scikit Learn.
HyperLearn also has statistical inference measures embedded, and can be called just like Scikit Learn's syntax (model.confidence_interval_)

## Key Methodologies and Aims
### Parallelism with CPU, GPUs
### Matrix Operation Tricks
### Matrix Decomposition & Inverse Tricks
### Fast Calculation of Extensive Statistical Inference Measures
### Fast & Drop In Module Deep Learning Integrations
### Clearer, Cleaner & Less Code Design
### Support for new and novel models

All functions are optimized as much as possible, using the following methodologies that I am currenlty researching on:
1. ### Parallelism with CPU, GPUs
  * Including Memory Sharing, Memory Management
  * CUDA Parallelism through PyTorch & Numba
2. ### Matrix Operation Tricks
  * Matrix Multiplication Ordering: https://en.wikipedia.org/wiki/Matrix_chain_multiplication
  * Element Wise Matrix Multiplication reducing complexity to O(n^2) from O(n^3): https://en.wikipedia.org/wiki/Hadamard_product_(matrices)
  * Reducing Matrix Operations to Einstein Notation: https://en.wikipedia.org/wiki/Einstein_notation
  * Evaluating one-time Matrix Operations in succession to reduce RAM overhead.
3. ### Matrix Decomposition & Inverse Tricks
  * If p>>n, maybe decomposing X.T is better than X.
  * Applying QR Decomposition then SVD might be faster in some cases.
  * Utilise the structure of the matrix to compute faster inverse (eg triangular matrices, Hermitian matrices).
  * Computing SVD(X) then getting pinv(X) is sometimes faster than pure pinv(X)
4. ### Fast Calculation of Extensive Statistical Inference Measures
  * Confidence, Prediction Intervals, Hypothesis Tests & Goodness of Fit tests for linear models are optimized.
  * Using Einstein Notation & Hadamard Products where possible.
  * Computing only what is neccessary to compute (Diagonal of matrix and not entire matrix).
  * Fixing the flaws of Statsmodels on notation, speed, memory issues and storage of variables.
5. ### Fast & Drop In Module Deep Learning Integrations
  * Using PyTorch to create Scikit-Learn like drop in replacements.
6. ### Clearer, Cleaner & Less Code Design
  * Using Decorators & Functions where possible.
  * Intuitive Middle Level Function names like (isTensor, isIterable).
  * Handles Parallelism easily through hyperlearn.multiprocessing
7. ### Support for new and novel models
  * Matrix Completion algorithms - Non Negative Least Squares, NNMF
  * Batch Similarity Latent Dirichelt Allocation (BS-LDA)
  * Correlation Regression
  * Feasible Generalized Least Squares FGLS
  * Outlier Tolerant Regression
  * Multidimensional Spline Regression
  * Generalized MICE (any model drop in replacement)
  * Using Uber's Pyro for Bayesian Deep Learning

A work in progress, HyperLearn aims to follow these development milestones:

## Goals & Development Schedule

### Will Focus on & why:
1. Singular Value Decomposition & QR Decomposition
 * SVD/QR is the backbone for many algorithms including:
   * Linear & Ridge Regression (Regression)
   * Statistical Inference for Regression methods (Inference)
   * Principal Component Analysis (Dimensionality Reduction)
   * Linear & Quadratic Discriminant Analysis (Classification & Dimensionality Reduction)
   * Pseudoinverse, Truncated SVD (Linear Algebra)
   * Latent Semantic Indexing LSI (NLP)
   * (new methods) Correlation Regression, FGLS, Outlier Tolerant Regression, Generalized MICE, Splines (Regression)

1. Port all important Numpy functions to faster alternatives (ONGOING)
  * Singular Value Decomposition (50% Complete)
    *
2. 

