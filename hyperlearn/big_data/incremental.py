
from numpy import vstack, newaxis, arange
from ..linalg import svd, eigh, eig
from .truncated import truncatedSVD, truncatedEigh
from ..utils import memoryXTX
from .randomized import randomizedSVD, randomizedEig
from ..exceptions import PartialWrongShape


def _utilSVD(batch, S, VT, eig = False):
	"""
	Batch (nrows, ncols)
	S (ncomponents)
	VT (rows = ncomponents, cols = ncols)
	
	Check Batch(ncols) == VT(ncols) to check same number
		of columns or else error is provided.
	"""
	if eig: 
		VT, S = VT.T, S**0.5
	ncomponents, ncols = VT.shape
	if batch.shape[1] != ncols:
		raise PartialWrongShape()

	data = vstack( ( S[:,newaxis]*VT , batch ) )

	return data, VT.shape[0] , memoryXTX(data)



def partialSVD(batch, S, VT, ratio = 1, solver = 'full', tol = None, max_iter = 'auto'):
	"""
	Fits a partial SVD after given old singular values S
	and old components VT.

	Note that VT will be used as the number of old components,
	so when calling truncated or randomized, will output a
	specific number of eigenvectors and singular values.

	Checks if new batch's size matches that of the old VT.

	Note that PartialSVD has different solvers. Either choose:
		1. full
			Solves full SVD on the data. This is the most
			stable and will guarantee the most robust results.
			You can select the number of components to keep
			within the model later.

		2. truncated
			This keeps the top K right eigenvectors and top
			k right singular values, as determined by
			n_components. Note full SVD is not called for the
			truncated case, but rather ARPACK is called.

		3. randomized
			Same as truncated, but instead of using ARPACK, uses
			randomized SVD.

	Notice how Batch = U @ S @ VT. However, partialSVD returns
	S, VT, and not U. In order to get U, you might consider using
	the relation that X = U @ S @ VT, and approximating U by:

		X = U @ S @ VT
		X @ V = U @ S
		(X @ V)/S = U

		So, U = (X @ V)/S, so you can output U from (X @ V)/S

		You can also get U partially and slowly using reverseU.
	"""
	data, k, __ = _utilSVD(batch, S, VT, eig = False)

	if solver == 'full':
		U, S, VT = svd(data)
	elif solver == 'truncated':
		U, S, VT = truncatedSVD(data, n_components = k, tol = tol)
	else:
		U, S, VT = randomizedSVD(data, n_components = k, max_iter = max_iter)

	return U[k:,:k], S[:k], VT[:k]




def partialEig(batch, S2, V, ratio = 1, solver = 'full', tol = None, max_iter = 'auto'):
	"""
	Fits a partial Eigendecomp after given old eigenvalues S2
	and old eigenvector components V.

	Note that V will be used as the number of old components,
	so when calling truncated or randomized, will output a
	specific number of eigenvectors and eigenvalues.

	Checks if new batch's size matches that of the old V.

	Note that PartialEig has different solvers. Either choose:
		1. full
			Solves full Eigendecompsition on the data. This is the most
			stable and will guarantee the most robust results.
			You can select the number of components to keep
			within the model later.

		2. truncated
			This keeps the top K right eigenvectors and top
			k eigenvalues, as determined by n_components. Note full Eig
			is not called for the truncated case, but rather ARPACK is called.

		3. randomized
			Same as truncated, but instead of using ARPACK, uses
			randomized Eig.
	"""
	data, k, memCheck = _utilSVD(batch, S2, V, eig = True)

	if solver == 'full':
		S2, V = eig(data, svd = False)
		return S2, V

	elif solver == 'truncated':
		if memCheck:
			S2, V = truncatedEigh(data.T @ data, n_components = k, tol = tol)
		else:
			__, S2, V = truncatedSVD(data, n_components = k, tol = tol)
			S2**=2
			V = V.T
	else:
		S2, V = randomizedEig(data, n_components = k, max_iter = max_iter)

	return S2[:k], V[:,:k]

