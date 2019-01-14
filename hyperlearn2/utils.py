
from .numba.types import *

@jit([ (void)(M_32, I64), (void)(M_64, I64) ], **nogil)  
def L_process(A, MIN):
    for i in range(MIN):
        A[i, i+1:] = 0
        A[i, i] = 1

