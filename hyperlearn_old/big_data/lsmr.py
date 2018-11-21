
from ..numba import norm, _min, _max, _sign
from numpy import infty, zeros, float32, float64
from copy import copy
from ..utils import _float


def Orthogonalize(a, b):
	A,B,_a,_b = abs(a), abs(b), _sign(a), _sign(b)
	if b == 0: 
		return _a, 0, A
	elif a == 0:
		return 0, _b, B
	elif B > A:
		tau = a/b
		s = _b/(1+tau*tau)**0.5
		c = s*tau
		r = b/s
	else:
		tau = b/a
		c = _a/(1+tau*tau)**0.5
		s = c*tau
		r = a/c
	return c,s,r


def floatType(dtype):
	dtype = str(dtype)
	if '64' in dtype:
		return float64
	return float32



def lsmr(X, y, tol = 1e-6, condition_limit = 1e8, alpha = 0, threshold = 1e12, non_negative = False,
			max_iter = None):
	"""
	[As of 12/9/2018, an optional non_negative argument is added. Note as accurate as Scipy's NNLS,
	by copies ideas from gradient descent.]

	Implements extremely fast least squares LSMR using orthogonalization as seen in Scipy's LSMR and
	https://arxiv.org/abs/1006.0758 [LSMR: An iterative algorithm for sparse least-squares problems]
	by David Fong, Michael Saunders.

	Scipy's version of LSMR is surprisingly slow, as some slow design factors were used
	(ie np.sqrt(1 number) is slower than number**0.5, or min(a,b) is slower than using 1 if statement.)

	ALPHA is provided for regularization purposes like Ridge Regression.

	This algorithm works well for Sparse Matrices as well, and the time complexity analysis is approx:
		X.T @ y   * min(n,p) times + 3 or so O(n) operations
		==> O(np)*min(n,p)
		==> either min(O(n^2p + n), O(np^2 + n))

	This complexity is much better than Cholesky Solve which is the next fastest in HyperLearn.
	Cholesky requires O(np^2) for XT * X, then Cholesky needs an extra 1/3*O(np^2), then inversion
	takes another 1/3*(np^2), and finally (XT*y) needs O(np).

	So Cholesky needs O(5/3np^2 + np) >> min(O(n^2p + n), O(np^2 + n))

	So by factor analysis, expect LSMR to be approx 2 times faster or so.
	Interestingly, the Space Complexity is even more staggering. LSMR takes only maximum O(np^2) space
	for the computation of XT * y + some overhead.

	Cholesky requires XT * X space, which is already max O(n^2p) [which is huge].
	Essentially, Cholesky shines when P is large, but N is small. LSMR is good for large N, medium P
	"""
	damp = alpha
	check = False
	try:
		X = X.A  # Convert Matrix to Array
		X = _float(X)
	except:
		pass
	dtype = X.dtype
	Y = y.ravel().copy()
	if y.dtype != dtype:
		Y = Y.astype(dtype)
	# Y = Y.copy()
	n,p = X.shape
	max_iter = _min(n, p) if max_iter is None else max_iter

	norm_Y = norm(Y)
	zero = zeros(p, dtype = dtype)
	theta_hat = zero.copy()

	beta = copy(norm_Y)
	XT = X.T

	if beta > 0:
		Y /= beta
		V = XT @ Y
		alpha = norm(V)
	else:
		V = zeros(p, dtype = dtype)
		alpha = 0

	if alpha > 0:
		V /= alpha
		
	# Initialize first iteration variables
	zeta_bar = alpha * beta
	alpha_bar = alpha
	rho, rho_bar, c_bar, s_bar = 1,1,1,0
	H = V.copy()
	H_bar = zero.copy()


	# Initialize variables for estimation of ||r||.
	beta_dd = beta
	beta_d, rho_d_old, tau_tilde_old = 0,1,0
	theta_tilde, zeta, d = 0,0,0


	# Initialize variables for estimation of ||A|| and cond(A)
	norm_X2 = alpha*alpha
	max_r_bar, min_r_bar = 0, 1e100
	norm_X, cond_X, norm_theta = alpha,1,0


	# Early stopping
	if condition_limit > 0:
		c_tol = 1/condition_limit
	else:
		c_tol = 0
	norm_r = beta


	# Check if theta_hat == 0
	normAB = alpha*beta
	if normAB == 0:
		return theta_hat, True


	# Iteration Loop
	for i in range(max_iter):

		Y *= -alpha
		Y += X @ V
		beta = norm(Y)

		if beta > 0:
			Y /= beta
			V *= -beta
			V += XT @ Y
			alpha = norm(V)

			if alpha > 0:
				V /= alpha


		c_hat, s_hat, alpha_hat = Orthogonalize(alpha_bar, damp)

		# Plane rotation
		rho_old = rho
		c, s, rho = Orthogonalize(alpha_hat, beta)
		theta_new = s*alpha
		alpha_bar = c*alpha

		# Plane rotation
		rho_bar_old, zeta_old = rho_bar, zeta
		theta_bar, rho_temp = s_bar*rho, c_bar*rho

		c_bar, s_bar, rho_bar = Orthogonalize(c_bar*rho, theta_new)
		zeta = c_bar*zeta_bar
		zeta_bar *= -s_bar
				
		# Update H, H_hat, theta_hat
		H_bar *= -theta_bar*rho/(rho_old*rho_bar_old)
		H_bar += H
		theta_hat += (zeta/(rho*rho_bar))*H_bar

		# Just in case if coefficients exceed maximum
		theta_hat[abs(theta_hat) > threshold] = 0.0

		if non_negative:
			theta_hat[theta_hat < 0] = 0.0

		H *= -theta_new/rho
		H += V

		# Estimate of ||r||
		beta_acute, beta_check = c_hat*beta_dd, -s_hat*beta_dd

		# Apply rotation
		beta_hat = c*beta_acute
		beta_dd = -s*beta_acute

		# Apply rotation
		theta_tilde_old = theta_tilde
		c_tilde_old, s_tilde_old, rho_tilde_old = Orthogonalize(rho_old, theta_bar)
		theta_tilde = s_tilde_old*rho_bar

		rho_old = c_tilde_old*rho_bar
		beta_d = c_tilde_old*beta_hat - s_tilde_old*beta_d


		tau_tilde_old = (zeta_old - theta_tilde_old*tau_tilde_old)/rho_tilde_old
		d += beta_check*beta_check

		partial = beta_d - ((zeta - theta_tilde*tau_tilde_old)/rho_old)
		norm_r = (d+ partial*partial + beta_dd*beta_dd)**0.5

		# Estimate ||X||
		norm_X2 += beta*beta
		norm_X = norm_X2**0.5
		norm_X2 += alpha*alpha

		# Condition(X)
		max_r_bar = _max(max_r_bar, rho_bar_old)
		if i > 1:
			min_r_bar = _min(max_r_bar, rho_bar_old)
		cond_X = _max(max_r_bar, rho_temp)/_min(min_r_bar, rho_temp)

		# Estimate other stuff

		test_1 = norm_r/norm_Y
		X_r = norm_X * norm_r


		if X_r != 0:
			test_2 = abs(zeta_bar) / X_r							# norm_ar = abs(zeta_bar)
		else:
			test_2 = infty

		if (1+test_2 <= 1):
			break
		elif (test_2 <= tol):
			break
		

		test_3 = 1/cond_X
		check = (1+test_3 <= 1)

		if check:
			break
		elif (test_3 <= c_tol):
			break


		partial = 1+(norm_X*norm(theta_hat)/norm_Y)						# norm_theta = norm(theta)
		t1 = test_1/partial

		r_tol = tol*partial

		if (1+t1 <= 1):
			break
		elif (test_1 <= r_tol):
			break


	return theta_hat, not check
	# check is TRUE is good, FALSE is cond(X) is too large.


