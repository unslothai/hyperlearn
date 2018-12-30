
from cpython cimport bool as BOOL

cimport numpy as np
import numpy as np
np.import_array()

ctypedef np.ndarray ARRAY
ctypedef bint bool
ctypedef long long LONG


cdef double FLOAT32_EPS = np.finfo(np.float32).eps
cdef double FLOAT64_EPS = np.finfo(np.float64).eps

cdef LONG UINT_SIZE[3]
UINT_SIZE[:] = [
    <LONG> np.iinfo(np.uint8).max,
    <LONG> np.iinfo(np.uint16).max,
    <LONG> np.iinfo(np.uint32).max
]

cdef list UINT_DTYPES = [
    np.zeros(1, dtype = np.uint8),
    np.zeros(1, dtype = np.uint16),
    np.zeros(1, dtype = np.uint32),
    np.zeros(1, dtype = np.uint64)
]

cdef LONG INT_SIZE[3]
INT_SIZE[:] = [
    <LONG> np.iinfo(np.int8).max,
    <LONG> np.iinfo(np.int16).max,
    <LONG> np.iinfo(np.int32).max
]

cdef list INT_DTYPES = [
    np.zeros(1, dtype = np.int8),
    np.zeros(1, dtype = np.int16),
    np.zeros(1, dtype = np.int32),
    np.zeros(1, dtype = np.int64)
]


######
cpdef ARRAY uinteger(LONG i):
    cdef int j
    for j in range(3):
        if i <= UINT_SIZE[j]:
            break
    return UINT_DTYPES[j]

######
cpdef ARRAY integer(LONG i):
    cdef int j
    for j in range(3):
        if i <= INT_SIZE[j]:
            break
    return INT_DTYPES[j]


######
cpdef (LONG, LONG) dot_left_right(LONG n, LONG a_b, LONG b_c, LONG c) nogil:
    cdef LONG AB, AB_C, left
    # From left X = (AB)C
    AB = a_b*b_c  # First row of AB
    AB *= n       # n * AB rows
    AB_C = b_c*c  # First row of AB_C
    AB_C *= n     # n * AB_C rows
    left = AB + AB_C

    cdef LONG BC, A_BC, right
    # From right X = A(BC)
    BC = b_c*c    # First row of BC
    BC *= a_b     # a_b * first row BC
    A_BC = a_b*c  # First row of A_BC
    A_BC *= n     # n times
    right = BC + A_BC
    
    return left, right


######
cpdef min_(a, b):
    if a < b:   return a
    return b
    
######
cpdef max_(a, b):
    if a < b:   return b
    return a


######
cpdef double epsilon(ARRAY X):
    cdef int size = X.itemsize
    cdef double eps
    
    if size < 8:
        eps = 1000.0
        eps *= FLOAT32_EPS
    else:
        eps = 1000000.0
        eps *= FLOAT64_EPS
    return eps


######
cpdef (int, int) svd_lwork(BOOL isComplex_dtype, int byte, LONG n, LONG p):
    """
    Computes the work required for SVD (gesdd, gesvd)
    [Updated 20/12/18 Uses Numba for some microsecond saving]
    """
    cdef LONG MIN, MAX
    cdef LONG gesdd, gesvd
    cdef LONG a, b
    
    if n < p:
        MIN = n
        MAX = p
    else:
        MIN = p
        MAX = n

    # check memory usage for GESDD vs GESVD. Use one which is within memory limits.
    if isComplex_dtype:
        gesdd = (MIN + 3)*MIN  # updated from netlib
        gesvd = 2*MIN + MAX
    else:
        # min(n, p)*(6 + min(n,p)) + max(n, p)
        gesdd = (4*MIN + 7)*MIN   # updated from netlib
        a = 3*MIN + MAX
        b = 5*MIN
        gesvd = a if a > b else b

    gesdd *= byte; gesvd *= byte;
    gesdd >>= 20; gesvd >>= 20;
    return gesdd, gesvd


###
cpdef (int, int) eigh_lwork(BOOL isComplex_dtype, int byte, LONG n, LONG p):
    """
    Computes the work required for EIGH (syevr, syevd, heevr, heevd)
    SYEVD = 1 + 6n + 2n^2
    SYEVR = 26n
    HEEVD = 2n + n^2
    HEEVR = 2n
    [Updated 20/12/18 Uses Numba for some microsecond saving]
    """
    cdef double mult = 1.1 * <double> n
    cdef LONG heevd, heevr, syevd, syevr
    cdef LONG evd, evr, s
    
    if p >= mult:   s = n
    else:           s = p

    if isComplex_dtype:
        heevd = 2*s + s*s
        heevr = 2*s + 1
        evd, evr = heevd, heevr
    else:
        syevd = 1 + 6*s + 2*s*s
        syevr = 26*s
        evd, evr = syevd, syevr

    evd *= byte; evr *= byte;
    evd >>= 20; evr >>= 20;
    return evd, evr

