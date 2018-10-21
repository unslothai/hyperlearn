<a href="github.com/danielhanchen/hyperlearn/"><img src="Images/HyperLearn_Logo.png" alt="drawing" width="350"/></a>

*Faster, Leaner GPU Sklearn, Statsmodels written in PyTorch*

![GitHub issues](https://img.shields.io/github/issues/badges/shields.svg?style=popout-square)
![Github All Releases](https://img.shields.io/github/downloads/atom/atom/total.svg?style=popout-square)
![Depfu](https://img.shields.io/depfu/depfu/example-ruby.svg?style=popout-square) *Currently badges don't work --> will update later :)*


---

**50%+ Faster, 50%+ less RAM usage, GPU support re-written Sklearn, Statsmodels combo with new novel algorithms.**

HyperLearn is written completely in PyTorch, NoGil Numba, Numpy, Pandas, Scipy & LAPACK, and mirrors (mostly) Scikit Learn.
HyperLearn also has statistical inference measures embedded, and can be called just like Scikit Learn's syntax (model.confidence_interval_)

---

<img src="Images/Packages_Used_long.png" alt="drawing" width="800"/>

### Comparison of Speed / Memory

| Algorithm         |  n    |  p  | Time(s) |            | RAM(mb) |            | Notes                   |
| ----------------- | ----- | --- | ------- | ---------- | ------- | ---------- | ----------------------- |
|                   |       |     | Sklearn | Hyperlearn | Sklearn | Hyperlearn |                         |
| QDA (Quad Dis A)  |1000000| 100 |   54.2  |   *22.25*  |  2,700  |  *1,200*   | Now parallelized        |
| LinearRegression  |1000000| 100 |   5.81  |   *0.381*  |   700   |    *10*    | Guaranteed stable & fast|

Time(s) is Fit + Predict. RAM(mb) = max( RAM(Fit), RAM(Predict) )
 
I've also added some preliminary results for N = 5000, P = 6000
<img src="Images/Preliminary Results N=5000 P=6000.png" alt="drawing" width="500"/>

Since timings are not good, I have submitted 2 bug reports to Scipy + PyTorch:
1. EIGH very very slow --> suggesting an easy fix #9212 https://github.com/scipy/scipy/issues/9212
2. SVD very very slow and GELS gives nans, -inf #11174 https://github.com/pytorch/pytorch/issues/11174

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
  * Computing only what is necessary to compute (Diagonal of matrix and not entire matrix).
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
On Licensing:
HyperLearn is under a GNU v3 License. This means:
1. Commercial use is restricted. Only software with 0 cost can be released. Ie: no closed source versions are allowed.
2. Using HyperLearn must entail all of the code being avaliable to everyone who uses your public software.
3. HyperLearn is intended for academic, research and personal purposes. Any explicit commercialisation of the algorithms and anything inside HyperLearn is strictly prohibited.

HyperLearn promotes a free a just world. Hence, it is free to everyone, except for those who wish to commercialise on top of HyperLearn.
