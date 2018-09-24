
from numba import _min, _max, maximum, minimum


def intialize_NMF(X, n_components = 2, eps = 1e-6, init = 'nndsvd'):

	U, S, VT = randomizedSVD(X, n_components = n_components)

	dtype = U.dtype
	W, H = np.zeros(U.shape, dtype), np.zeros(VT.shape, dtype)
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