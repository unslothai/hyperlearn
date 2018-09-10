
from numpy import vstack, newaxis
from ..linalg import svd, eigh, eig
from .truncated import truncatedSVD, truncatedEigh
from ..utils import memoryXTX
from .randomized import randomizedSVD, randomizedEig


def _util(batch, S, VT, n_components = 1, ratio = 1, eig = False):
    if eig:
        VT = VT.T
    comp, col = VT.shape
    assert batch.shape[1] == col
    components = int(n_components*ratio)

    if components > col:  components = col

    assert comp == components
    
    if eig:
        data = vstack( ( (S**0.5)[:,newaxis] * VT  ,  batch ) )
    else:
        data = vstack( ( S[:,newaxis] * VT  ,  batch ) )

    return data, components, memoryXTX(data)



def partialSVD(batch, S, VT):
	"""
	Fits a partial SVD after given old singular values S
	and old components VT.

	Checks if new batch's size matches that of the old VT.
	"""
    data, __, memCheck = _util(batch, S, VT)

    if (data.shape[1] > data.shape[0]) or ~memCheck: 
        __, S, VT = svd( data, transpose = True )
    else:
        S, VT = eigh(data.T @ data, svd = True)
    return S, VT



def partialEig(batch, S2, V):
    """
    Fits a partial Eigendecomposition after given old eigenvalues S2
    and old components VT.

    Checks if new batch's size matches that of the old VT.
    """
    data, __, memCheck = _util(batch, S2, V, eig = True)

    if (data.shape[1] > data.shape[0]) or ~memCheck: 
        S2, V = eig(data)
    else:
        S2, V = eigh(data.T @ data, positive = True)
    return S2, V



def truncatedPartialSVD(batch, S, VT, n_components = 2, ratio = 2):
    """
    HyperLearn also extends Truncated SVD to fit on partial
    datasets. If n_components is set, then truncatedPartialSVD
    will output only n_components * ratio.

    Note the accuracy will be reduced when compared to full partialSVD,
    since the corrected eigenspace is not very robust.

    Thus, RATIO is set to 2, were the n_components outputted is
    actually 2 * n_components, and not n_components. If you deem
    that it's better to output purely n_components, then set
    ratio to 1.
    """
    data, components, memCheck = _util(batch, S, VT, n_components, ratio)

    if (data.shape[1] > data.shape[0]) or ~memCheck:
        __, S, VT = truncatedSVD( data, transpose = True, n_components = components )
    else:
        S, VT = truncatedEigh(data.T @ data, svd = True, n_components = components )
    return S, VT



def truncatedPartialEig(batch, S2, V, n_components = 2, ratio = 2):
    """
    HyperLearn also extends Truncated Eig to fit on partial
    datasets. If n_components is set, then truncatedPartialEig
    will output only n_components * ratio.

    Note the accuracy will be reduced when compared to full partialEig,
    since the corrected eigenspace is not very robust.

    Thus, RATIO is set to 2, were the n_components outputted is
    actually 2 * n_components, and not n_components. If you deem
    that it's better to output purely n_components, then set
    ratio to 1.
    """
    data, components, memCheck = _util(batch, S2, V, n_components, ratio, eig = True)

    if (data.shape[1] > data.shape[0]) or ~memCheck:
        __, S2, V = truncatedSVD( data, transpose = True, n_components = components )
        V = V.T
        S2 **= 2
    else:
        S2, V = truncatedEigh(data.T @ data, svd = False, n_components = components )
    return S2, V



def randomizedPartialSVD(batch, S, VT, n_components = 2, ratio = 3, max_iter = 'auto', solver = 'lu'):
    """
    HyperLearn also extends Randomized SVD to fit on partial
    datasets. If n_components is set, then randomizedPartialSVD
    will output only n_components * ratio.

    Note the accuracy will be reduced when compared to full partialSVD,
    since the corrected eigenspace is not very robust.

    Thus, RATIO is set to 3, were the n_components outputted is
    actually 3 * n_components, and not n_components. Note this is even
    more strict than truncatedPartialSVD, since randomizedSVD's
    P(success) is lower.
    """
    data, components, memCheck = _util(batch, S, VT, n_components, ratio)

    if (data.shape[1] > data.shape[0]) or ~memCheck:
        __, S, VT = randomizedSVD( data, n_components = components, max_iter = max_iter, solver = solver)
    else:
        S, VT = randomizedEig(data.T @ data, n_components = components, max_iter = max_iter, solver = solver)
    return S, VT



def randomizedPartialEig(batch, S2, V, n_components = 2, ratio = 3, max_iter = 'auto', solver = 'lu'):
    """
    HyperLearn also extends Randomized Eig to fit on partial
    datasets. If n_components is set, then randomizedPartialEig
    will output only n_components * ratio.

    Note the accuracy will be reduced when compared to full partialEig,
    since the corrected eigenspace is not very robust.

    Thus, RATIO is set to 3, were the n_components outputted is
    actually 3 * n_components, and not n_components. Note this is even
    more strict than truncatedPartialEig, since randomizedEig's
    P(success) is lower.
    """
    data, components, memCheck = _util(batch, S2, V, n_components, ratio, eig = True)

    if (data.shape[1] > data.shape[0]) or ~memCheck:
        __, S2, V = randomizedSVD( data, n_components = components, max_iter = max_iter, solver = solver)
        V = V.T
        S2 **= 2
    else:
        S2, V = randomizedEig(data.T @ data, n_components = components, max_iter = max_iter, solver = solver)
    return S2, V

