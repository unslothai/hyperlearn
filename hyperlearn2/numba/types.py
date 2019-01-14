
from numba.types import float32, float64, int64, int8, bool_, uint32, int_, void
from numba.types import Tuple
from numba import jit, prange

nogil = {"fastmath":True,"nogil":True,"cache":True,"nopython":True}
gil = {"fastmath":True,"nogil":False,"cache":True,"nopython":True}
parallel = {"fastmath":True,"nogil":True,"cache":True,"nopython":True,"parallel":True}

M32 = float32[:,:]
M64 = float64[:,:]
A32 = float32[::1]
A64 = float64[::1]

M32_ = float32[:,::1]
M64_ = float64[:,::1]

M_32 = float32[::1,:]
M_64 = float64[::1,:]

I64 = int64
F32 = float32
F64 = float64
U32 = uint32
I = int_

