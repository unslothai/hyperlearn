%%cython
#cython:boundscheck=False,wraparound=False,language_level=3,initializedcheck=False,cdivision=True, nonecheck = False
cimport numpy as np
import numpy as np
np.import_array()
from libc.stdlib cimport malloc, free
ctypedef np.ndarray ARRAY
ctypedef bint bool
ctypedef (long long) LONG

cdef inline int INT = sizeof(int)
cdef inline int FLOAT = sizeof(float)
cdef inline int DOUBLE = sizeof(double)

from scipy.linalg.cython_lapack cimport slacpy, dlacpy

#--------------    
cdef np.ndarray[float,ndim=2] copy_float_(
    np.ndarray[float,ndim=2] X, int n, int p, bool Fcontiguous, bool overwrite):
    cdef char UPLO = ord("A")
    cdef int* _n
    cdef np.ndarray[float,ndim=2] out
    
    if not Fcontiguous:
        out = np.empty((n,p), dtype = np.float32, order = "C")
        _n = &p
        slacpy(&UPLO, _n, &n, &X[0,0], _n, &out[0,0], _n)
        if overwrite:  del X
        return out.T
    elif not overwrite:
        out = np.empty((n,p), dtype = np.float32, order = "F")
        _n = &n
        slacpy(&UPLO, _n, &p, &X[0,0], _n, &out[0,0], _n)
        return out
    return X
    
#--------------
cdef copy_float(X, n, p, Fcontiguous, overwrite):
    if n*p <= 4000: 
        if not Fcontiguous or not overwrite: return np.array(X, copy = True, order = "F")
        return X
    return copy_float_(X, n, p, Fcontiguous, overwrite)

#--------------    
cdef np.ndarray[double,ndim=2] copy_double_(
    np.ndarray[double,ndim=2] X, int n, int p, bool Fcontiguous, bool overwrite):
    cdef char UPLO = ord("A")
    cdef int* _n
    cdef np.ndarray[double,ndim=2] out
    
    if not Fcontiguous:
        out = np.empty((n,p), dtype = np.float64, order = "C")
        _n = &p
        dlacpy(&UPLO, _n, &n, &X[0,0], _n, &out[0,0], _n)
        if overwrite:  del X
        return out.T
    elif not overwrite:
        out = np.empty((n,p), dtype = np.float64, order = "F")
        _n = &n
        dlacpy(&UPLO, _n, &p, &X[0,0], _n, &out[0,0], _n)
        return out
    return X
    
#--------------
cdef copy_double(X, n, p, Fcontiguous, overwrite):
    if n*p <= 4000: 
        if not Fcontiguous or not overwrite: return np.array(X, copy = True, order = "F")
        return X
    return copy_double_(X, n, p, Fcontiguous, overwrite)

#--------------    
cdef np.ndarray[float,ndim=2] symmetric_float(np.ndarray[float,ndim=2] X, int n, int* _n):
    cdef char UPLO = ord("U")
    cdef np.ndarray[float,ndim=2] out = np.empty((n,n), dtype = np.float32, order = "F")
    slacpy(&UPLO, _n, _n, &X[0,0], _n, &out[0,0], _n)
    return out

#--------------    
cdef np.ndarray[double,ndim=2] symmetric_double(np.ndarray[double,ndim=2] X, int n, int* _n):
    cdef char UPLO = ord("U")
    cdef np.ndarray[double,ndim=2] out = np.empty((n,n), dtype = np.float64, order = "F")
    dlacpy(&UPLO, _n, _n, &X[0,0], _n, &out[0,0], _n)
    return out


#--------------#--------------
from scipy.linalg.cython_lapack cimport dgesdd as _dgesdd, sgesdd as _sgesdd

#--------------
cpdef (int,int,int,int,int,int,bool,bool,bool,int) gesdd_work(np.ndarray X, bool overwrite = False):
    cdef:
        int n = X.shape[0], p = X.shape[1], itemsize = X.itemsize
        int MIN, MAX, temp, iwork_, lwork = 0
        LONG memory = 0
        bool tall = (n >= p), Fcontiguous = X.flags.f_contiguous
    if tall:
        MIN, MAX = p, n
    else:
        MIN, MAX = n, p
    
    memory = MIN + MIN*MIN # overwrite and S use same memory for 2 parts
    # Notice - contents of A are destroyed.
    if not Fcontiguous:
        # Force overwrite to save memory since array anyways copied
        overwrite = True
        memory += n*p
    if overwrite:
        # 3*mn + max( mx, 5*mn*mn + 4*mn )
        temp = 5*MIN*MIN + 4*MIN
        lwork = temp if temp > MAX else MAX
        lwork += 3*MIN
    else:
        # 4*mn*mn + 7*mn
        lwork = 4*MIN*MIN + 7*MIN
        memory += n*p
    memory += lwork
    iwork_ = INT*8*MIN
    memory = (memory*itemsize + iwork_)>>20 # also + Iwork
    # 1.15 factor from experimental checks
    return n, p, lwork, iwork_, MIN, MAX, tall, Fcontiguous, overwrite, <int>(<double>memory*1.15)


#--------------
cpdef dgesdd(
    np.ndarray[double,ndim=2] X, int n, int p, int lwork, int iwork_, int MIN, int MAX, bool tall,
    bool Fcontiguous, bool overwrite):
    cdef:
        np.ndarray[double,ndim=2] U, VT, A = copy_double(X, n, p, Fcontiguous, overwrite)
        double* U_pointer = NULL
        double* VT_pointer = NULL
        np.ndarray[double,ndim=1] S = np.empty(MIN, dtype = np.float64, order = 'F')
        int info
        char jobz
        int* _n = &n

    #-------------- Create U, VT
    if not overwrite:
        jobz = ord('S')
        U = np.empty((n, MIN), dtype = np.float64, order = 'F')
        VT = np.empty((MIN, p), dtype = np.float64, order = 'F')
        U_pointer, VT_pointer = &U[0,0], &VT[0,0]
    elif tall: # U is overwritten on A
        jobz = ord("O")
        VT = np.empty((p, p), dtype = np.float64, order = 'F')
        VT_pointer = &VT[0,0]
    else: # VT is overwritten on A
        jobz = ord("O")
        U = np.empty((n, n), dtype = np.float64, order = 'F')
        U_pointer = &U[0,0]
        
    cdef int* iwork = <int*> malloc(iwork_)
    cdef double* work = <double*> malloc(lwork*DOUBLE)
    #-------------- Call LAPACK
    _dgesdd(&jobz, _n, &p, &A[0,0], _n, &S[0], U_pointer, _n, VT_pointer, 
            &MIN, work, &lwork, iwork, &info)
    
    #-------------- Return and clean up malloced space
    free(work); free(iwork);
    if not overwrite: return U, S, VT
    elif tall:        return A, S, VT
    else:             return U, S, A
    
    
#--------------
cpdef sgesdd(
    np.ndarray[float,ndim=2] X, int n, int p, int lwork, int iwork_, int MIN, int MAX, bool tall,
    bool Fcontiguous, bool overwrite):
    cdef:
        np.ndarray[float,ndim=2] U, VT, A = copy_float(X, n, p, Fcontiguous, overwrite)
        float* U_pointer = NULL
        float* VT_pointer = NULL
        np.ndarray[float,ndim=1] S = np.empty(MIN, dtype = np.float32, order = 'F')
        int info
        char jobz
        int* _n = &n

    #-------------- Create U, VT
    if not overwrite:
        jobz = ord('S')
        U = np.empty((n, MIN), dtype = np.float32, order = 'F')
        VT = np.empty((MIN, p), dtype = np.float32, order = 'F')
        U_pointer, VT_pointer = &U[0,0], &VT[0,0]
    elif tall: # U is overwritten on A
        jobz = ord("O")
        VT = np.empty((p, p), dtype = np.float32, order = 'F')
        VT_pointer = &VT[0,0]
    else: # VT is overwritten on A
        jobz = ord("O")
        U = np.empty((n, n), dtype = np.float32, order = 'F')
        U_pointer = &U[0,0]
    
    
    cdef int* iwork = <int*> malloc(iwork_)
    cdef float* work = <float*> malloc(lwork*FLOAT)
    #-------------- Call LAPACK
    _sgesdd(&jobz, _n, &p, &A[0,0], _n, &S[0], U_pointer, _n, VT_pointer, 
            &MIN, work, &lwork, iwork, &info)
    
    #-------------- Return and clean up malloced space
    free(work); free(iwork);
    if not overwrite: return U, S, VT
    elif tall:        return A, S, VT
    else:             return U, S, A    
    
#--------------#--------------
from scipy.linalg.cython_lapack cimport sgesvd as _sgesvd, dgesvd as _dgesvd

#--------------
cpdef (int,int,int,char,char,int,bool,bool,bool,bool,int) gesvd_work(
    np.ndarray X, bool overwrite = False, bool VT_only = False):
    cdef:
        int n = X.shape[0], p = X.shape[1], itemsize = X.itemsize        
        int MIN, MAX, temp1, temp2, lwork
        LONG memory = 0
        bool tall = (n >= p), Fcontiguous = X.flags.f_contiguous
        char JOBU, JOBVT
    if tall:
        MIN, MAX = p, n
    else:
        MIN, MAX = n, p
        
    temp1 = 3*MIN+MAX; temp2 = 5*MIN
    lwork = temp1 if temp1 > temp2 else temp2
    
    memory = MIN + lwork
    if not Fcontiguous or (Fcontiguous and not overwrite): memory += n*p
    
    if VT_only:
        JOBU, JOBVT = ord("N"), ord("O")
    else: # Need the other part
        if tall: JOBU, JOBVT = ord("O"), ord("S")
        else: JOBVT, JOBU = ord("O"), ord("S")
        memory += MIN*MIN
    
    memory = (memory*itemsize)>>20
    return n, p, MIN, JOBU, JOBVT, lwork, tall, Fcontiguous, overwrite, VT_only, memory


#--------------
cpdef sgesvd(
    np.ndarray[float,ndim=2] X, int n, int p, int MIN, char JOBU, char JOBVT,
    int lwork, bool tall, bool Fcontiguous, bool overwrite, bool VT_only):
    cdef:
        np.ndarray[float,ndim=2] U, VT, A = copy_float(X, n, p, Fcontiguous, overwrite)
        float* U_pointer
        float* VT_pointer
        np.ndarray[float,ndim=1] S = np.empty(MIN, dtype = np.float32, order = "F")
        int info
        int* _n = &n
    
    #-------------- Create U, VT
    if VT_only: pass
    elif tall: # Overwrite A with U
        VT = np.empty((p, p), dtype = np.float32, order = 'F')
        VT_pointer = &VT[0,0]
    else:
        U = np.empty((n, n), dtype = np.float32, order = 'F')
        U_pointer = &U[0,0]
    
    cdef float* work = <float*> malloc(FLOAT*lwork)
    #-------------- Call LAPACK
    _sgesvd(&JOBU, &JOBVT, _n, &p, &A[0,0], _n, &S[0], U_pointer, _n,
            VT_pointer, &MIN, work, &lwork, &info)
    
    #-------------- Return and clean up malloced space
    free(work);
    if VT_only:     return S, A[:MIN]
    elif tall:      return A, S, VT
    else:           return U, S, A
    
    
#--------------
cpdef dgesvd(
    np.ndarray[double,ndim=2] X, int n, int p, int MIN, char JOBU, char JOBVT,
    int lwork, bool tall, bool Fcontiguous, bool overwrite, bool VT_only):
    cdef:
        np.ndarray[double,ndim=2] U, VT, A = copy_double(X, n, p, Fcontiguous, overwrite)
        double* U_pointer
        double* VT_pointer
        np.ndarray[double,ndim=1] S = np.empty(MIN, dtype = np.float64, order = "F")
        int info
        int* _n = &n
    
    #-------------- Create U, VT
    if VT_only: pass
    elif tall: # Overwrite A with U
        VT = np.empty((p, p), dtype = np.float64, order = 'F')
        VT_pointer = &VT[0,0]
    else:
        U = np.empty((n, n), dtype = np.float64, order = 'F')
        U_pointer = &U[0,0]
    
    #-------------- Call LAPACK
    cdef double* work = <double*> malloc(DOUBLE*lwork)
    _dgesvd(&JOBU, &JOBVT, _n, &p, &A[0,0], _n, &S[0], U_pointer, _n,
            VT_pointer, &MIN, work, &lwork, &info)
    
    #-------------- Return and clean up malloced space
    free(work);
    if VT_only:     return S, A[:MIN]
    elif tall:      return A, S, VT
    else:           return U, S, A


#--------------#--------------
from scipy.linalg.cython_lapack cimport sgetrf, dgetrf
from scipy.linalg.cython_lapack cimport slaswp, dlaswp
from .utils import L_process

#--------------
cpdef (int,int,int,bool,bool,bool,int) getrf_work(
    np.ndarray[float,ndim=2] X, bool overwrite = False):
    cdef:
        int n = X.shape[0], p = X.shape[1], itemsize = X.itemsize     
        LONG memory = 0
        bool tall = (n >= p), Fcontiguous = X.flags.f_contiguous
        int MIN = p if tall else n
        
    if not Fcontiguous:   memory = n*p
    elif not overwrite:   memory = n*p
        
    memory += (MIN*INT + memory*itemsize)>>20 # include pivot
    return n, p, MIN, tall, Fcontiguous, overwrite, memory
    
        
#--------------
cpdef np.ndarray[float,ndim=2] sgetrf_L_only(np.ndarray[float,ndim=2] X, 
    int n, int p, int MIN, bool tall, bool Fcontiguous, bool overwrite = False):
    cdef:
        np.ndarray[float,ndim=2] A = copy_float(X, n, p, Fcontiguous, overwrite)
        int info, k1 = 1
        int* _n = &n
        int* _MIN = &MIN
        float* _A = &A[0,0]
        int* _info = &info
        int* pivot = <int*> malloc(INT*MIN)
        
    sgetrf(_n, &p, _A, _n, pivot, _info)
    
    if not tall: # wide needs to be small
        A.resize((n,MIN), refcheck = False) # Frees columns and removes memory
    
    L_process(A, MIN) # keep L
    
    info = -1
    slaswp(_MIN, _A, _n, &k1, _MIN, pivot, _info)
    
    free(pivot)
    return A

#--------------
cpdef np.ndarray[double,ndim=2] dgetrf_L_only(np.ndarray[double,ndim=2] X, 
    int n, int p, int MIN, bool tall, bool Fcontiguous, bool overwrite = False):
    cdef:
        np.ndarray[double,ndim=2] A = copy_float(X, n, p, Fcontiguous, overwrite)
        int info, k1 = 1
        int* _n = &n
        int* _MIN = &MIN
        double* _A = &A[0,0]
        int* _info = &info
        int* pivot = <int*> malloc(INT*MIN)
        
    dgetrf(_n, &p, _A, _n, pivot, _info)
    
    if not tall: # wide needs to be small
        A.resize((n,MIN), refcheck = False) # Frees columns and removes memory
    
    L_process(A, MIN) # keep L
    
    info = -1
    dlaswp(_MIN, _A, _n, &k1, _MIN, pivot, _info)
    
    free(pivot)
    return A


#--------------#--------------
from scipy.linalg.cython_lapack cimport sgeqrf as _sgeqrf, dgeqrf as _dgeqrf
from scipy.linalg.cython_lapack cimport sorgqr, dorgqr
from scipy.linalg.cython_lapack cimport sormqr, dormqr

#--------------
cpdef (int,int,int,bool,int,bool,bool,int) geqrf_work(np.ndarray X, bool overwrite = False):
    cdef:
        int n = X.shape[0], p = X.shape[1], itemsize = X.itemsize
        bool tall = (n>=p)
        int MIN = p if tall else n
        int lwork
        LONG memory = 0
        bool Fcontiguous = X.flags.f_contiguous
        
    lwork = 3*n
    if not Fcontiguous:   memory = n*p
    elif not overwrite:   memory = n*p
    memory += ((lwork+memory + MIN)*itemsize)>>20  # Include TAU array
    return n, p, MIN, tall, lwork, Fcontiguous, overwrite, memory


#--------------
cpdef np.ndarray[float,ndim=2] sgeqrf_Q_only(np.ndarray[float,ndim=2] X, 
    int n, int p, int MIN, bool tall, int lwork, bool Fcontiguous, bool overwrite):
    cdef:
        np.ndarray[float,ndim=2] A = copy_float(X, n, p, Fcontiguous, overwrite)
        int info
        int* _n = &n
        int* _p = &p
        int* _lwork = &lwork
        int* _info = &info
        float* _A = &A[0,0]
        float* tau = <float*> malloc(FLOAT*MIN)
        float* work = <float*> malloc(FLOAT*lwork)
    
    _sgeqrf(_n, _p, _A, _n, tau, work, _lwork, _info)
    if not tall: # wide needs to be small
        A.resize((n, MIN), refcheck = False) # Frees columns and removes memory
    
    sorgqr(_n, _p, &MIN, _A, _n, tau, work, _lwork, _info)

    free(tau); free(work);
    return A


#--------------
cpdef np.ndarray[double,ndim=2] dgeqrf_Q_only(np.ndarray[double,ndim=2] X, 
    int n, int p, int MIN, bool tall, int lwork, bool Fcontiguous, bool overwrite):
    cdef:
        np.ndarray[double,ndim=2] A = copy_double(X, n, p, Fcontiguous, overwrite)
        int info
        int* _n = &n
        int* _p = &p
        int* _lwork = &lwork
        int* _info = &info
        double* _A = &A[0,0]
        double* tau = <double*> malloc(DOUBLE*MIN)
        double* work = <double*> malloc(DOUBLE*lwork)
    
    _dgeqrf(_n, _p, _A, _n, tau, work, _lwork, _info)
    if not tall: # wide needs to be small
        A.resize((n, MIN), refcheck = False) # Frees columns and removes memory
    
    dorgqr(_n, _p, &MIN, _A, _n, tau, work, _lwork, _info)

    free(tau); free(work);
    return A


# from scipy.linalg.cython_lapack cimport sgeqrt3, dgeqrt3
# from scipy.linalg.cython_lapack cimport sgemqrt

# cpdef np.ndarray[float,ndim=2] sgeqrt(
#     np.ndarray[float,ndim=2] X, bool overwrite = False, np.ndarray[float,ndim=2] C):
#     cdef:
#         int n = X.shape[0], p = X.shape[1]
#         np.ndarray[float,ndim=2] A = copy_float(X, n, p, X.flags.f_contiguous, overwrite)
#         float[::1,:] T = np.empty((p,p), order = "F", dtype = np.float32)
#         int info

#     sgeqrt3(&n, &p, &A[0,0], &n, &T[0,0], &p, &info)
    
#     cdef np.ndarray[float,ndim=2] out = np.empty((p,C.shape[0]), dtype = np.float32)
    
#     sgemqrt()
    
#     del T
#     del A
#     return out

from scipy.linalg.cython_lapack cimport ssyevr as _ssyevr, dsyevr as _dsyevr

#--------------
cpdef (int,int,int,bool,int) syevr_work(np.ndarray X, bool overwrite = False):
    cdef:
        int n = X.shape[0], itemsize = X.itemsize
        int lwork, liwork
        LONG memory = n # eigenvalues
        
    lwork = 26*n
    liwork = 10*n
    if not overwrite:   memory += n*n
    memory += (lwork*itemsize +(liwork+2*n)*INT)>>20  # Include IWORK array + ISUPPZ
    return n, lwork, liwork, overwrite, memory

#--------------
cpdef ssyevr(np.ndarray[float,ndim=2] X, int n, int lwork, int liwork, bool overwrite):
    cdef:
        np.ndarray[float,ndim=1] W = np.empty(n, dtype = np.float32, order = "F")
        np.ndarray[float,ndim=2] V = np.empty((n,n), dtype = np.float32, order = "F")        
        char JOBZ = ord("V"), RANGE = ord("A"), UPLO = ord("U")
        float abstol = 0
        int info
        int* _info = &info
        int* _n = &n
        float* work = <float*> malloc(FLOAT*lwork)
        int* iwork = <int*> malloc(INT*liwork)
        int* ISUPPZ = <int*> malloc(2*n*INT)
        np.ndarray[float,ndim=2] A = X if overwrite else symmetric_float(X, n, _n)

    _ssyevr(&JOBZ, &RANGE, &UPLO, _n, &A[0,0], _n, NULL, NULL, NULL, NULL,
           &abstol, _info, &W[0], &V[0,0], _n, ISUPPZ, work, &lwork, iwork, &liwork, _info)
    del A
    free(work); free(iwork); free(ISUPPZ);
    return W, V


#--------------
cpdef dsyevr(np.ndarray[double,ndim=2] X, int n, int lwork, int liwork, bool overwrite):
    cdef:
        np.ndarray[double,ndim=1] W = np.empty(n, dtype = np.float64, order = "F")
        np.ndarray[double,ndim=2] V = np.empty((n,n), dtype = np.float64, order = "F")        
        char JOBZ = ord("V"), RANGE = ord("A"), UPLO = ord("U")
        double abstol = 0
        int info
        int* _info = &info
        int* _n = &n
        double* work = <double*> malloc(DOUBLE*lwork)
        int* iwork = <int*> malloc(INT*liwork)
        int* ISUPPZ = <int*> malloc(2*n*INT)
        np.ndarray[double,ndim=2] A = X if overwrite else symmetric_double(X, n, _n)

    _dsyevr(&JOBZ, &RANGE, &UPLO, _n, &A[0,0], _n, NULL, NULL, NULL, NULL,
           &abstol, _info, &W[0], &V[0,0], _n, ISUPPZ, work, &lwork, iwork, &liwork, _info)
    del A
    free(work); free(iwork); free(ISUPPZ);
    return W, V


#--------------#--------------
from scipy.linalg.cython_lapack cimport ssyevd as _ssyevd, dsyevd as _dsyevd

#--------------
cpdef (int,int,int,bool,int) syevd_work(np.ndarray X, bool overwrite = False):
    cdef:
        int n = X.shape[0], itemsize = X.itemsize
        int lwork, liwork
        LONG memory = n # eigenvalues
        
    lwork = 1 + 6*n + 2*n*n
    liwork = 3 + 5*n
    if not overwrite:   memory += n*n
    memory += (lwork*itemsize +liwork*INT)>>20  # Include IWORK array + ISUPPZ
    return n, lwork, liwork, overwrite, memory


#--------------
cpdef ssyevd(np.ndarray[float,ndim=2] X, int n, int lwork, int liwork, bool overwrite):
    cdef:
        np.ndarray[float,ndim=1] W = np.empty(n, dtype = np.float32, order = "F")       
        char JOBZ = ord("V"), UPLO = ord("U")
        int info
        int* _n = &n
        float* work = <float*> malloc(FLOAT*lwork)
        int* iwork = <int*> malloc(INT*liwork)
        np.ndarray[float,ndim=2] A 
        float* A_pointer
        
    if overwrite: A_pointer = &X[0,0]
    else: 
        A = symmetric_float(X, n, _n)
        A_pointer = &A[0,0]

    _ssyevd(&JOBZ, &UPLO, _n, A_pointer, _n, &W[0], work, &lwork, iwork, &liwork, &info)
    free(work); free(iwork);
    if overwrite:  return W, X
    else:          return W, A

