Hey Contributor!

Thanks for checking out HyperLearn!! Super appreciate it.

Since the package is new (only started like August 27th)..., Issues are the best place to start helping out, and or check out the Projects tab. There's a whole list of stuff I envisioned to complete.

Also, if you have a NEW idea: please post an issue and label it new enhancement.

In terms of priorities, I wanted to start from the bottom up, as to make all functions faster; that means:

Since Singular Value Decomp is the backbone for nearly all Linear algos (PCA, Linear, Ridge reg, LDA, QDA, LSI, Partial LS, etc etc...), we need to focus on making SVD faster! (Also Linear solvers).

When SVD optimization is OK, then slowly creep into Least Squares / L1 solvers. These need to be done before other algorithms in order for speedups to be apparent.

If NUMBA code is used, it needs to be PRE-COMPILED in order to save time, or else we need to wait a whopping 2-3 seconds before each call...

Then, focus on Linear Algorithms, including but not limited to:

Linear Regression
Ridge Regression
SVD Solving, QR Solving, Cholesky Solving for backend
Linear Discriminant Analysis
Quadratic Discriminant Analysis
Partial SVD (Truncated SVD --> maybe used Facebook's PCA? / Gensim's LSI? --> try not to use ARPACK's svds...)
Full Principal Component Analysis (complete SVD decomp)
Partial PCA (Truncated SVD)
Canonical Correlation Analysis
Partial Least Squares
Spline Regression based on Least Squares (will talk more later on this)
Correlation Regression (will talk more later on this)
Outlier Tolerant Regression (will talk more later on this)
SGDRegressor / SGDCLassifier easily through PyTorch
Batch Least Squares? --> Closed Form soln + averaging
Others...
