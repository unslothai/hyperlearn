from .base import *
from scipy.stats import t as tdist
from .numba import mean

__all__ = ['corr', 'qr_stats','svd_stats','ridge_stats']


def corr(X, y):
    same = y is X
    
    _X = X - mean(X, 0)
    _y = y - mean(y, 0) if not same else _X
    
    if len(X.shape) == 1:
        _X2 = einsum('i,i', _X, _X)**0.5
    else:
        _X2 = einsum('ij,ij->j', _X, _X)**0.5
        

    if len(y.shape) == 1:
        _y2 = einsum('i,i',_y,_y)**0.5
    else:
        _y2 = einsum('ij,ij->j',_y,_y)**0.5 if not same else _X2
        _X2 = _X2[:,newaxis]
        
    corr = (_X.T @ _y) / _X2 /_y2
    
    return corr

"""
------------------------------------------------------------
QR_STATS
Updated 27/8/2018
------------------------------------------------------------
"""
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

"""
------------------------------------------------------------
SVD_STATS
Updated 27/8/2018
------------------------------------------------------------
"""
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


"""
------------------------------------------------------------
RIDGE_STATS
Updated 27/8/2018
------------------------------------------------------------
"""
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
