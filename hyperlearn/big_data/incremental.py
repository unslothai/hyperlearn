
from numpy import vstack, newaxis, arange
from ..linalg import svd, eigh, eig
from .truncated import truncatedSVD, truncatedEigh
from ..utils import memoryXTX
from .randomized import randomizedSVD, randomizedEig
from ..exceptions import PartialWrongShape


def _util(batch, S, VT, eig = False):
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




def reverseU(X, S, VT, size = 1):
	"""
	Computes an approximation to the matrix U if given S and VT and
	the data matrix X. If size = 1, then will not batch process.
	Else, if size > 1, then will split the data into portions
	and find U approximation.
	
	U is approximated by:

		X = U @ S @ VT
		X @ V = U @ S
		(X @ V)/S = U

		So, U = (X @ V)/S, so you can output U from (X @ V)/S
	"""
	V = VT.T
	if size > 1:
		n = len(X)
		parts = list(arange(0, n, int(n/size)))
		if parts[-1] != n:
			parts.append(n)
		
		Us = []
		for left,right in zip(parts, parts[1:]):
			Us.append( (X[left:right]@V)/S )
		return Us
			
	return (X @ V)/S




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

		You can also get U partially and slowly using batchU.
	"""
	data, k, memCheck = _util(batch, S, VT, eig = False)

	if solver == 'full':
		S, VT = eig(batch, svd = True)
		return S, VT

	elif solver == 'truncated':
		S, VT = truncatedSVD(batch, n_components = k, tol = tol)
	else:
		S, VT = randomizedSVD(batch, n_components = k, max_iter = max_iter)

	return S[:k], VT[:k]




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
	

