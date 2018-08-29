from .base import *
from torch import svd as __svd, qr as __qr
from scipy.linalg.lapack import clapack
from torch import potrf as cholesky_decomposition, diag, ones, \
                potrs as cholesky_triangular_solve

__all__ = ['svd','_svd','pinv','_pinv','qr_solve','svd_solve',
			'ridge_solve','squareSum','rowSum',
            'cholesky_solve','cholesky_decomposition']

"""
------------------------------------------------------------
QR_SOLVE
Updated 27/8/2018
------------------------------------------------------------
"""
def t_qr(X):
    return __qr(X)
qr = n2n(t_qr)
_qr = n2t(t_qr)


def qr_solve(X, y):
    '''
    theta =  R^-1 * QT * y
    '''
    Q, R = qr(X)
    check = 0
    if R.shape[0] == R.shape[1]:
        _R, check = clapack.strtri(R)
    if check > 0:
        _R = _pinv(R)
    Q, _R, R = Tensor(Q, _R, R)
    
    theta_hat = _R.matmul(   T(Q).matmul( ravel(y, Q) )   )
    return theta_hat


"""
------------------------------------------------------------
SVD_SOLVE & PINV
Updated 27/8/2018
------------------------------------------------------------
"""

def t_svd(X, U = True):
    if U:
        U, S, V = __svd(X, some = True)
        return U, S, T(V)
    else:
        __, S, V = __svd(X, some = True)
        return S, T(V)
svd = n2n(t_svd)
_svd = n2t(t_svd)


def t_pinv(X):
    U, S, VT = _svd(X)
    cond = S < eps(X)*constant(S[0])
    _S = 1.0 / S
    _S[cond] = 0.0
    VT *= T(_S)
    return T(VT).matmul(T(U))
pinv = n2n(t_pinv)
_pinv = n2t(t_pinv)


def svd_solve(X, y):
    '''
    theta =  V * S^-1 * UT * y
    '''
    U, S, VT = _svd(X)
    cond = S < eps(X)*constant(S[0])
    _S = 1.0 / S
    _S[cond] = 0.0
    VT *= T(_S)
    
    theta_hat = T(VT).matmul(  
                            T(U).matmul(  ravel(y, U)  )
                            )
    return theta_hat


"""
------------------------------------------------------------
RIDGE_SOLVE using SVD
Updated 27/8/2018
------------------------------------------------------------
"""

def ridge_solve(X, y, alpha = 1):
    '''
                    S
    theta =   V --------- UT y 
                 S^2 + aI
    '''
    U, S, VT = _svd(X)
    cond = S < eps(X)*constant(S[0])
    _S = S / (S**2 + alpha)
    _S[cond] = 0.0
    VT *= T(_S)
    
    theta_hat = T(VT).matmul(  
                            T(U).matmul(  ravel(y, U)  )
                            )
    return theta_hat

"""
------------------------------------------------------------
CHOLESKY_SOLVE
Updated 29/8/2018
------------------------------------------------------------
"""
def t_cholesky_solve(X, y, alpha = 0):
    '''
    Solve least squares problem X*theta_hat = y using Cholesky Decomposition.
    
    |  Method   |   Operations    | Factor * np^2 |
    |-----------|-----------------|---------------|
    | Cholesky  |   1/3 * np^2    |      1/3      |
    |    QR     |   p^3/3 + np^2  |   1 - p/3n    |
    |    SVD    |   p^3   + np^2  |    1 - p/n    |
    
    NOTE: HyperLearn's implementation of Cholesky Solve uses L2 Regularization to enforce stability.
    Cholesky is known to fail on ill-conditioned problems, so adding L2 penalties helpes it.
    
    Note, the algorithm in this implementation is as follows:
    
        alpha = dtype(X).decimal    [1e-6 is float32]
        while failure {
            solve cholesky ( XTX + alpha*identity )
            alpha *= 10
        }
    
    If MSE (Mean Squared Error) is abnormally high, it might be better to solve using stabler but
    slower methods like qr_solve, svd_solve or lstsq.
    
    https://www.quora.com/Is-it-better-to-do-QR-Cholesky-or-SVD-for-solving-least-squares-estimate-and-why
    '''

    XTX = T(X).matmul(X)
    regularizer = ones(X.shape[1]).type(X.dtype)
    
    if alpha == 0: 
        alpha = typeTensor([np_finfo(dtype(X)).resolution]).type(X.dtype)
    no_success = True
    warn = False

    while no_success:
        alphaI = regularizer*alpha
        try:
            chol = cholesky_decomposition(  XTX + diag(alphaI)  )
            no_success = False
        except:
            alpha *= 10
            warn = True
            
    if warn and print_all_warnings:
        addon = constant(alpha.round(10))
        print(f'''
            Matrix is ill-conditioned. Added regularization = {addon} to combat this. 
            Now, solving L2 regularized (XTX+{addon}*I)^-1(XTy).

            NOTE: It might be better to use svd_solve, qr_solve or lstsq if MSE is high.
            ''')
   
    XTy = T(X).matmul( ravel(y, chol)  )
    
    theta_hat = cholesky_triangular_solve(XTy, chol).flatten()
    return theta_hat

cholesky_solve = n2n(t_cholesky_solve)
_cholesky_solve = n2t(t_cholesky_solve)

