
from numpy import nanmean, nanstd, log1p, isnan, sqrt, nanmin, nan
from ..numba import minimum
from ..linalg import eig
from ..big_data.randomized import randomizedEig
from ..big_data.incremental import partialSVD


def fit(X, n_components = 'auto', standardise = True, copy = True):
	"""
	[Added 31/10/2018] [Edited 2/11/2018 Fixed SVDImpute]

	Fits a SVD onto the training data by projecting it to a lower space after
	being intially filled with column means. By default, n_components is
	determined automatically using log(p+1). Setting too low or too high mirrors
	mean imputation, and deletes the purpose of SVD imputation.

	Returns:
	1. S 	singular values
	2. VT 	eigenvectors
	+ mean, std, mins
	"""
	n, p = X.shape
	k = int(sqrt(p)-1) if n_components in ('auto', None) else n_components        
	if k <= 0: k = 1
	if k >= p: k = p

	C = X.copy() if copy else X
	mask = isnan(X)

	if standardise:
		mean = nanmean(X, 0)
		std = nanstd(X, 0)
		mins = nanmin(X, 0)
		std[std == 0] = 1
		C -= mean
		C /= std
	else:
		mean, std, mins = None, None, None
	C[mask] = 0

	S, VT = randomizedEig(C, k)
	S **= 0.5
	VT = VT.T

	if copy == False:
		C[mask] = nan
		if standardise:
			C *= std
			C += mean
	return S, VT, mean, std, mins, standardise


def transform(X, S, VT, mean, std, mins, standardise, copy = True):
	"""
	[Added 31/10/2018] [Edited 2/11/2018 FIxed SVDImpute]

	The fundamental advantage of HyperLearn's SVD imputation is that a .transform
	method is provided. I do not require seeing the whole matrix for imputation,
	and can calculate SVD incrementally via the Incremental Module.
	"""
	n, p = X.shape
	D = X.copy() if copy else X
	mask = isnan(X)

	if standardise:
		D -= mean
		D /= std
	D[mask] = 0
	
	U, S, VT = partialSVD(D, S, VT, solver = 'randomized')
	reconstruction = U * S @ VT
	D[mask] = reconstruction[mask]

	if standardise:
		D *= std
		D += mean
		for j in range(p):
			min_ = mins[j]
			what = D[:,j]
			what[what < min_] = min_

	if copy == False:
		X[mask] = nan
		if standardise:
			X *= std
			X += mean
	return D
