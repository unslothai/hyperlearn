
from . import numba
from .utils import *
import numpy as np
from . import linalg


_reflect = reflect
def corr(X, y, reflect = False):
    """
    Provides the Pearson Correlation between X and y. y can be X,
    and if that's the case, will use faster methods. Much faster
    than Scipy's version, as HyperLearn is parallelised.
    [Added 8/12/18]

    Parameters
    -----------
    X:              Correlate y with this
    y:              The main matrix or vector you want to correlate with
    reflect:        Whether to output full correlations, or 1/2 filled
                    upper triangular matrix. Default = False.
    Returns
    -----------
    C:              Correlation matrix.
    """
    same = y is X
    
    _X = X - numba.mean(X, 0)
    _y = y - numba.mean(y, 0) if not same else _X
    
    # Sometimes norm can be = 0
    _X2 = numba.maximum( col_norm(_X)**0.5, 1)
    
    if len(y.shape) == 1:
        _y2 = col_norm(_y)**0.5
        if _y2 == 0:
            _y2 = 1
    else:
        _y2 = col_norm(_y)**0.5 if not same else _X2
        _y2 = numba.maximum(_y2, 1)
        _X2 = _X2[:,np.newaxis]
        
    if same:
        C = linalg.matmul("X.H @ X", _X)
    else:        
        C = _X.T @ _y
    C /= _X2
    C /= _y2
    if same and reflect:
        C = _reflect(C)
    return C


@jit
def corr_sum(C):
    """
    Sums up all abs(correlation values). Used to find the
    most "important" columns.
    """
    p = C.shape[0]
    z = np.zeros(p, dtype = C.dtype)

    for i in range(p):
        for j in range(i+1, p):
            c = abs(C[i,j])
            z[i] += c
            z[j] += c            
    return z
