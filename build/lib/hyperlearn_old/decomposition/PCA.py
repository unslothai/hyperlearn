
from .base import _basePCA
from ..base import *
from ..linalg import svd, eigh, svd_flip


class PCA(_basePCA):
    """
    Principal Component Analysis reduces the dimensionality
    of data by selecting the linear combinations of columns
    or features which maximises the total variance.
    
    Speed
    --------------
    Note, SKLEARN's implementation is wasteful, as it uses
    a full SVD solver, which is slow and painful.
    
    HyperLearn's implementation uses regularized Eigenvalue Decomposition,
    but if solver = 'svd', then full SVD is used.
    
    If USE_GPU:
        Uses PyTorch's SVD (which is slow sadly), or EIGH. Speed is OK.
    If CPU:
        Uses Numpy's Fortran C based SVD or EIGH.
        If NUMBA is not installed, uses very fast LAPACK functions.
    
    Stability
    --------------
    Alpha is added for regularization purposes. This prevents system
    rounding errors and promises better convergence rates.
    
    Note svd_flip is NOT same as SKLEARN, hence output may have reversed
    signs. V based decision is used as EIGH is faster, and U is not computed.
    """
    def __init__(self, n_components = 2, solver = 'eig', alpha = None, fast = True,
                centre = True):
        self.decomp = self._fit_svd if solver == 'svd' else self._fit_eig
        
        self.n_components, self.alpha, self.solver, self.fast, self.truncated, \
        self.centre    = n_components, alpha, solver, fast, False, centre
        
        
    def _fit_svd(self, X):        
        S2, VT = eig(X, fast = self.fast)
        return S2, VT
        
        
    def _fit_eig(self, X):
        if X.shape[1] >= X.shape[0]:
            # Drop back to SVD, as Eigendecomp would output U and not VT.
            return self._fit_svd(X)
        else:
            S2, VT = eigh(X.T @ X, alpha = self.alpha, fast = self.fast,
                            positive = True)
            return S2, VT
