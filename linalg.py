from .base import *
from torch import svd as __svd, qr as __qr
from scipy.linalg.lapack import clapack
from scipy.stats import t as tdist

__all__ = ['svd','_svd','pinv','_pinv','qr_solve','svd_solve',
			'ridge_solve','qr_stats','svd_stats','ridge_stats',
            'squareSum','rowSum']

def squareSum(X):
    return einsum('ij,ij->i', X, X )

def rowSum(X, Y = None):
    if Y is None:
        return einsum('ij->i',X)
    return einsum('ij,ij->i', X , Y )


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


def qr_stats(Q, R):
    '''
    XTX^-1  =  RT * R
    
    h = diag  Q * QT
    
    mean(h) used for normalized leverage
    '''
    XTX = T(R).matmul(R)
    _XTX = pinv(XTX)
    ## Einsum is slow in pytorch so revert to numpy version
    h = squareSum(Q) #einsum('ij,ij->i', Q, Q )
    h_mean = h.mean()
    
    return _XTX, h, h_mean


def svd_stats(U, S, VT):
    '''
                  1
    XTX^-1 =  V ----- VT 
                 S^2
    
    h = diag U * UT
    
    mean(h) used for normalized leverage
    '''
    _S2 = 1.0 / (S**2)
    VS = T(VT) * _S2
    _XTX = VS.matmul(VT)
    h = squareSum(U) #einsum('ij,ij->i', U, U )
    h_mean = h.mean()
    
    return _XTX, h, h_mean


def ridge_stats(U, S, VT, alpha = 1):
    '''
                               S^2
    exp_theta_hat =  diag V --------- VT
                            S^2 + aI
                            
                                 S^2
    var_theta_hat =  diag V ------------- VT
                            (S^2 + aI)^2
    
                    1
    XTX^-1 =  V --------- VT
                S^2 + aI
                
                  S^2
    h = diag U --------- UT
                S^2 + aI
    
    mean(h) used for normalized leverage
    '''
    V = T(VT)
    S2 = S**2
    S2_alpha = S2 + alpha
    S2_over_S2 = S2 / S2_alpha
    
    VS = V * S2_over_S2
    exp_theta_hat = einsum('ij,ji->i', VS, VT )     # Same as VS.dot(VT)
    
    V_S2 = VS / S2_alpha  # np_divide(S2,  np_square( S2 + alpha ) )
    var_theta_hat = rowSum(V_S2, V) #einsum('ij,ij->i',  V_S2  , V )   # Sams as np_multiply(   V,   V_S2 ).sum(1)
    
    _XTX = (V * (1.0 / S2_alpha )  ).matmul( VT )   # V  1/S^2 + a  VT
    
    h = rowSum((U * S2_over_S2), U) #einsum('ij,ij->i', (U * S2_over_S2), U )  # Same as np_multiply(  U*S2_over_S2, U ).sum(1)
    h_mean = h.mean()
    
    return exp_theta_hat, var_theta_hat, _XTX, h_mean

