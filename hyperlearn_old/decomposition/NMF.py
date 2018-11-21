
from ..numba import _min, _max, maximum, minimum, norm, njit, prange, squaresum
from numpy import zeros, float32, float64
from ..utils import _float, reflect, _XTX, _XXT
from ..big_data.randomized import randomizedSVD
from ..solvers import solveCholesky


def intialize_NMF(X, n_components = 2, eps = 1e-6, init = 'nndsvd', HT = True):
	U, S, VT = randomizedSVD(X, n_components = n_components)

	dtype = U.dtype
	W, H = zeros(U.shape, dtype), zeros(VT.shape, dtype)
	Sa = S[0]**0.5

	W[:, 0] = Sa * abs(U[:, 0])
	H[0, :] = Sa * abs(VT[0, :])

	for j in range(1, n_components):
		a, b = U[:,j], VT[j,:]

		a_p, b_p = maximum(a, 0), maximum(b, 0)
		a_n, b_n = abs(minimum(a, 0)), abs(minimum(b, 0))

		a_p_norm, b_p_norm = norm(a_p), norm(b_p)
		a_n_norm, b_n_norm = norm(a_n), norm(b_n)

		m_p, m_n = a_p_norm * b_p_norm, a_n_norm * b_n_norm

		# Update
		if m_p > m_n:
			a_p /= a_p_norm
			b_p /= b_p_norm
			u,v,sigma = a_p, b_p, m_p
		else:
			a_n /= a_n_norm
			b_n /= b_n_norm
			u,v,sigma = a_n, b_n, m_n

		lbd = (S[j] * sigma)**0.5
		W[:,j], H[j,:] = lbd*u, lbd*v

	W, H = maximum(W, 0), maximum(H, 0)
	if HT:
		return W, H.T.copy(), X
	return W, H, X



def update_CD_base(W, HHT, XHT, n, k, runs = 1):
	violation = 0
	XHT *= -1
	
	for t in prange(k):
		# Hessian
		H_part = HHT[t]
		hess = H_part[t]

		if hess == 0:
			for run in range(runs):
				for i in prange(n):
					W_i = W[i]
					W_it = W_i[t]
					# gradient = GW[t, i] where GW = np.dot(W, HHt) - XHt
					grad = XHT[i, t]

					for r in prange(k): grad += H_part[r] * W_i[r]
					
					# projected gradient
					pg = _min(0., grad) if W_it == 0 else grad
					violation += abs(pg)

		else:
			for run in range(runs):
				for i in prange(n):
					W_i = W[i]
					W_it = W_i[t]
					# gradient = GW[t, i] where GW = np.dot(W, HHt) - XHt
					grad = XHT[i, t]

					for r in prange(k): grad += H_part[r] * W_i[r]
					
					# projected gradient
					pg = _min(0., grad) if W_it == 0 else grad
					violation += abs(pg)
					
					if grad != 0:
						W[i, t] = _max(W_it - grad / hess, 0.)
	return violation
update_CD = njit(update_CD_base, fastmath = True, nogil = True, cache = True)
update_CD_parallel = njit(update_CD_base, fastmath = True, nogil = True, parallel = True)



def nmf_cd(X, n_components = 2, tol = 1e-4, max_iter = 200, init = 'nndsvd', speed = 1, n_jobs = 1):
	W, HT, X = intialize_NMF(X, n_components)

	XT = X.T
	n,k = W.shape
	p,k = HT.shape

	update_CD_i = update_CD_parallel if n_jobs != 1 else update_CD

	if speed != 1: 
		max_iter = _min(int(200/speed*1.5), 5)

	for n_iter in range(max_iter):
		# Update W
		#HHT = reflect()
		violation = update_CD_i(W, HT.T@HT, X@HT, n, k, speed)
		# Update H
		violation += update_CD_i(HT, W.T@W, XT@W, p, k, speed)
		#loss.append(squareSum(X - W@HT.T).sum())
		
		if n_iter == 0:
			violation_init = violation

		if violation_init == 0:
			break

		if violation / violation_init <= tol:
			break
	return W, HT.T



def nmf_als(X, n_components = 2, max_iter = 100, init = 'nndsvd', alpha = None):
	W, H, X = intialize_NMF(X, n_components, HT = False)
	XT = X.T
	n = X.shape[0]
	past_error = 1e100

	for i in range(max_iter):
		H = maximum(solveCholesky(W, X, alpha = alpha), 0)
		W = maximum(solveCholesky(H.T, XT, alpha = alpha), 0).T
		if i % 10 == 0:
			error = squaresum(X - W@H)/n
			if error/past_error > 0.9:
				break
			past_error = error
	return W, H



_X = zeros((2,2), float32)
_XX = nmf_cd(_X, 1, max_iter = 1)
_X = nmf_cd(_X, 1, max_iter = 1, n_jobs = -1)
_X = zeros((2,2), float64)
_XX = nmf_cd(_X, 1, max_iter = 1)
_X = nmf_cd(_X, 1, max_iter = 1, n_jobs = -1)
_X = None
_XX = None
