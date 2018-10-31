
from numpy import nan, nanmean, nanstd
from ..numba import minimum, isnan
from ..linalg import eig
from ..big_data.randomized import randomizedSVD
from ..big_data.truncated import truncatedSVD
from ..big_data.incremental import partialSVD


def determine_k(p, k = None):
	if k == None:
		k = int(log1p(k))
		if k <= 0: k = 1
	if type(k) == float: k *= p
	k = int(k) if k < p else p
	return k


def fit(X, n_components = None, solver = 'randomized'):
	x_train = X.copy()
	n, p = x_train
	k = determine_k(p, n_components)	

	mean = nanmean(x_train, 0)
	std = minimum(nanstd(x_train, 0), 1)

	x_train[isnan(x_train)] = 0
	x_train -= mean
	x_train /= std
	
	if solver == 'randomized':
		S, VT = randomizedEig(x_train, k)
		S **= 0.5
		VT = V.T
	elif solver == 'truncated':
		__, S, VT = truncatedSVD(x_train, k)
	else:
		S, VT = eig(x_train, svd = True)
		S, VT = S[:k], VT[:k]
	return S, VT


def transform(X, S, VT, solver = 'randomized'):
	x_test = X.copy()
	x_test[isnan(x_test)] = 0
	x_test -= mean
	x_test /= std

	U, S, VT = partialSVD(x_test, S, VT, solver)
	reconstruct = U * S @ VT
	reconstruct *= std
	reconstruct += mean

	return reconstruct


