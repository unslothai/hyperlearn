
include "DEFINE.pyx"
from libc.stdlib cimport rand, RAND_MAX
from libc.math cimport log, sqrt

cdef UINT32 cycle = RAND_MAX
cdef UINT64 cycle2 = RAND_MAX*RAND_MAX
cdef float divisor = <float> (RAND_MAX + 3)

######
# XOR is assosciative. So can do in any order. So precalculate
#
# cdef SIZE j
# cdef UINT64 boolean_mask = 1
# cdef UINT64 shifter = 1
# for j in range(32):
#     boolean_mask |= shifter
#     shifter <<= 2
cdef UINT64 boolean_mask = 6148914691236517205

######
cdef void _bool(UINT8[:] out, SIZE size, UINT64 x) nogil:
    cdef SIZE i, j, k = 0
    cdef UINT64 shift
    cdef SIZE div = size // 64
    cdef SIZE diff = size % 64
    
    for i in range(div):
        x = 16807 * x + 2531011
        # Use XOR gate to invert some bits
        x ^= boolean_mask
        shift = 1
        for j in range(64):
            out[k] = ((x & shift) == 0); k += 1; shift <<= 1
            
    x, shift = 16807 * x + 2531011, 1
    for j in range(diff):
        out[k] = ((x & shift) == 0); k += 1; shift <<= 1

######
cdef void bool_np(UINT8[:,::1] out, SIZE n, SIZE p, int seed) nogil:
    cdef SIZE r, i, j, k
    cdef UINT64 x = <UINT64> (rand() if seed < 0 else seed)
    cdef UINT64 shift = <UINT64> (2531011*x + 214013)
    cdef UINT64 mask
    cdef SIZE reduce = p-1
    cdef SIZE div = reduce // 64
    cdef SIZE diff = reduce % 64

    for r in range(n):
        k = 1
        for i in range(div):
            x = 16807 * x + 2531011
            # Use XOR gate to invert some bits
            x ^= boolean_mask
            mask = 1
            for j in range(64):
                out[r, k] = ((x & mask) == 0); k += 1; mask <<= 1
                    
        x, mask = 16807 * x + 2531011, 1
        for j in range(diff):
            out[r, k] = ((x & mask) == 0); k += 1; mask <<= 1








    if isTuple:
        out2D = 
        with nogil:
            for i in range(n): _bool(out2D[i], p, shift + 3*<UINT32>i)
            _bool(out2D[:,0], n, x)
    else:
        out = np.zeros(n, dtype = np.uint8)
        _bool(out, n, x)
    return (np.asarray(out2D) if isTuple else np.asarray(out)).view(np.bool_)


######
cdef void uint8_(UINT8[::1] out, UINT32 x, float mult, SIZE div, SIZE diff) nogil:
    cdef SIZE i, j = 0
    for i in range(div):
        x = (8121 * x + 12345)&cycle;   out[j] = <UINT8> (x*mult+shift); j+=1
        x = (x^0x6EEE)&cycle;           out[j] = <UINT8> (x*mult+shift); j+=1
        x = (x^0xABCD)&cycle;           out[j] = <UINT8> (x*mult+shift); j+=1
        x = (x^0xCCCC)&cycle;           out[j] = <UINT8> (x*mult+shift); j+=1
    if diff >= 1:   x = (8121 * x + 12345)&cycle; out[j] = <UINT8> (x*mult+shift); j+=1
    if diff >= 2:   x = (x^0x6EEE)&cycle; out[j] = <UINT8> (x*mult+shift); j+=1
    if diff >= 3:   x = (x^0xABCD)&cycle; out[j] = <UINT8> (x*mult+shift); j+=1

######
cdef void _uint16(UINT16[::1] out, UINT32 x, float mult, SIZE div, SIZE diff) nogil:
    cdef SIZE i, j = 0
    for i in range(div):
        x = (65793 * x + 28411)&cycle;  out[j] = <UINT16> (x*mult+shift); j+=1
        x = (x^0x6EEE)&cycle;           out[j] = <UINT16> (x*mult+shift); j+=1
        x = (x^0xABCD)&cycle;           out[j] = <UINT16> (x*mult+shift); j+=1
        x = (x^0xCCCC)&cycle;           out[j] = <UINT16> (x*mult+shift); j+=1
    if diff >= 1:   x = (65793 * x + 28411)&cycle; out[j] = <UINT16> (x*mult+shift); j+=1
    if diff >= 2:   x = (x^0x6EEE)&cycle; out[j] = <UINT16> (x*mult+shift); j+=1
    if diff >= 3:   x = (x^0xABCD)&cycle; out[j] = <UINT16> (x*mult+shift); j+=1

######
cdef void _uint32(UINT32[::1] out, UINT32 x, float mult, SIZE div, SIZE diff) nogil:
    cdef SIZE i, j = 0
    for i in range(div):
        x = (214013 * x +2531011)&cycle;out[j] = <UINT32> (x*mult+shift); j+=1
        x = (x^0x6EEE)&cycle;           out[j] = <UINT32> (x*mult+shift); j+=1
        x = (x^0xABCD)&cycle;           out[j] = <UINT32> (x*mult+shift); j+=1
        x = (x^0xCCCC)&cycle;           out[j] = <UINT32> (x*mult+shift); j+=1
    if diff >= 1:   x = (214013 * x + 2531011)&cycle; out[j] = <UINT32> (x*mult+shift); j+=1
    if diff >= 2:   x = (x^0x6EEE)&cycle; out[j] = <UINT32> (x*mult+shift); j+=1
    if diff >= 3:   x = (x^0xABCD)&cycle; out[j] = <UINT32> (x*mult+shift); j+=1

######
cdef void _uint64(UINT64[::1] out, UINT64 x, double mult, SIZE div, SIZE diff) nogil:
    cdef SIZE i, j = 0
    for i in range(div):
        x = (214013 * x+2531011)&cycle2;out[j] = <UINT64> (x*mult+shift); j+=1
        x = (x^0x6EEE)&cycle2;          out[j] = <UINT64> (x*mult+shift); j+=1
        x = (x^0xABCD)&cycle2;          out[j] = <UINT64> (x*mult+shift); j+=1
        x = (x^0xCCCC)&cycle2;          out[j] = <UINT64> (x*mult+shift); j+=1
    if diff >= 1:   x = (214013 * x + 2531011)&cycle2; out[j] = <UINT64> (x*mult+shift); j+=1
    if diff >= 2:   x = (x^0x6EEE)&cycle2; out[j] = <UINT64> (x*mult+shift); j+=1
    if diff >= 3:   x = (x^0xABCD)&cycle2; out[j] = <UINT64> (x*mult+shift); j+=1

######
cdef void _float32(float[::1] out, UINT32 x, double mult, SIZE div, SIZE diff) nogil:
    cdef SIZE i, j = 0
    for i in range(div):
        x = (214013 * x+ 2531011)&cycle;out[j] = (<float>x*mult+shift); j+=1
        x = (x^0x6EEE)&cycle;           out[j] = (<float>x*mult+shift); j+=1
        x = (x^0xABCD)&cycle;           out[j] = (<float>x*mult+shift); j+=1
        x = (x^0xCCCC)&cycle;           out[j] = (<float>x*mult+shift); j+=1
    if diff >= 1:   x = (214013 * x + 2531011)&cycle; out[j] = (<float>x*mult+shift); j+=1
    if diff >= 2:   x = (x^0x6EEE)&cycle; out[j] = (<float>x*mult+shift); j+=1
    if diff >= 3:   x = (x^0xABCD)&cycle; out[j] = (<float>x*mult+shift); j+=1

######
cdef void _float64(double[::1] out, UINT32 x, double mult, SIZE div, SIZE diff) nogil:
    cdef SIZE i, j = 0
    for i in range(div):
        x = (214013 * x+2531011)&cycle2;out[j] = (<double>x*mult+shift); j+=1
        x = (x^0x6EEE)&cycle2;          out[j] = (<double>x*mult+shift); j+=1
        x = (x^0xABCD)&cycle2;          out[j] = (<double>x*mult+shift); j+=1
        x = (x^0xCCCC)&cycle2;          out[j] = (<double>x*mult+shift); j+=1
    if diff >= 1:   x = (214013 * x + 2531011)&cycle2; out[j] = (<double>x*mult+shift); j+=1
    if diff >= 2:   x = (x^0x6EEE)&cycle2; out[j] = (<double>x*mult+shift); j+=1
    if diff >= 3:   x = (x^0xABCD)&cycle2; out[j] = (<double>x*mult+shift); j+=1


##############
cdef (float, float, UINT32, UINT32) process_float(
    float left, float right, int seed, UINT32 a, UINT32 b) nogil:

    cdef float low, high, add, mult, shift
    cdef UINT32 x, change

    add = <float> fabs(left)
    if left < 0:
        low, high = 0, right + add
    else:
        low, high = left, right

    mult = (high - low)/divisor
    shift = <float>(left if left >= 0 else (low - add))
    x = <UINT32> (rand() if seed < 0 else seed)
    change = <UINT32> (a * x + b)
    return mult, shift, x, change
    

######
cdef ARRAY float32_(
    float left, float right, SIZE n, SIZE p, int seed, SIZE div, SIZE diff, bool isTuple):
    cdef float[::1] out
    cdef float[:,::1] out2D
    cdef SIZE i
    cdef float mult, shift
    cdef UINT32 x, change
    mult, shift, x, change = process_float(left, right, seed, 214013, 2531011)

    if isTuple:
        out2D = np.zeros((n,p), dtype = np.float32)
        with nogil:
            for i in range(n): 
                _float32(out2D[i], change+<UINT32>i, mult, shift, div, diff)
    else:
        out = np.zeros(n, dtype = np.float32)
        _float32(out, x, mult, shift, div, diff)

    return np.asarray(out2D) if isTuple else np.asarray(out)


##############
cdef (double, double, UINT64, UINT64) process_double(double left, double right, int seed) nogil:
    cdef double low, high, add, mult, shift
    cdef UINT64 x, change

    add = <double> fabs(left)
    if left < 0:
        low, high = 0, right + add
    else:
        low, high = left, right

    mult = (high - low)/divisor2
    shift = <double>(left if left >= 0 else (low - add))
    x = <UINT64> (rand() if seed < 0 else seed)
    change = <UINT64> (214013 * x + 2531011)
    return mult, shift, x, change


######
cdef ARRAY int64_(
    int left, int right, SIZE n, SIZE p, int seed, SIZE div, SIZE diff, bool isTuple):
    cdef INT64[::1] out
    cdef INT64[:,::1] out2D
    cdef SIZE i
    cdef double mult, shift
    cdef UINT64 x, change
    mult, shift, x, change = process_double(left, right, seed)

    if isTuple:
        out2D = np.zeros((n,p), dtype = np.int64)
        with nogil:
            for i in range(n): 
                _int64(out2D[i], change+<UINT64>i, mult, shift, div, diff)
    else:
        out = np.zeros(n, dtype = np.int64)
        _int64(out, x, mult, shift, div, diff)

    return np.asarray(out2D) if isTuple else np.asarray(out)


######
cdef ARRAY float64_(
    double left, double right, SIZE n, SIZE p, int seed, SIZE div, SIZE diff, bool isTuple):
    cdef double[::1] out
    cdef double[:,::1] out2D
    cdef SIZE i
    cdef double mult, shift
    cdef UINT64 x, change
    mult, shift, x, change = process_double(left, right, seed)

    if isTuple:
        out2D = np.zeros((n,p), dtype = np.float64)
        with nogil:
            for i in range(n): 
                _float64(out2D[i], change+<UINT64>i, mult, shift, div, diff)
    else:
        out = np.zeros(n, dtype = np.float64)
        _float64(out, x, mult, shift, div, diff)

    return np.asarray(out2D) if isTuple else np.asarray(out)


#######
cdef (SIZE,SIZE,SIZE,SIZE,bool,char) args_process(dtype, size, int factor):
    cdef char dt = ord(np.dtype(dtype).char)
    cdef SIZE div, n, p
    cdef SIZE diff

    cdef type S = type(size)
    cdef bool isTuple = S is tuple or S is list or S is np.ndarray

    if isTuple:
        if len(size) == 1:
            n, p = 0, 0
        else:
            n, p = <SIZE> size[0], <SIZE> size[1]
        if n <= 0 or p <= 0: raise AssertionError("Size must be > 0")
        div, diff = p // factor, p % factor
    else:
        n, p = <SIZE>size, 0
        if n <= 0: raise AssertionError("Size must be > 0")
        div, diff = n // factor, n % factor
    return div, n, p, diff, isTuple, dt


######
cpdef ARRAY uniform(
    double left = 0, double right = 10, size = 10, int seed = -1, dtype = np.float32):
    """
    Creates normal uniform numbers. Uses ideas from the Box Muller Transfrom,
    and inspired from (senderle)'s Stackoverflow implementation of fast
    normal numbers. Also uses a modified Linear Congruential Generator for
    uniform numbers. There are also 2 modes (slow, fast). Slow is more accurate.
    Fast approximates sqrt(-2log(x)/x) as sqrt(2)(1/(x+0.03)-0.8)
    """
    
    if left >= right: raise AssertionError("Right must be > left")

    cdef char dt
    cdef SIZE div, n, p, diff
    cdef bool isTuple
    div, n, p, diff, isTuple, dt = args_process(dtype, size, 4)

    cdef int l, r
    l, r = <int> fabs(left), <int> fabs(right)
    cdef int _max = l if l > r else r
    l, r = <int> left, <int> right

    if dt == boolean:
        return bool_(n, p, seed, isTuple)

    ###
    if dt == uint8:
        if left < 0: l = 0
        if right > uint8_max: dt = uint16
        else: return uint8_(l, r, n, p, seed, div, diff, isTuple)

    if dt == uint16:
        if left < 0: l = 0
        if right > uint16_max: dt = uint32
        else: return uint16_(l, r, n, p, seed, div, diff, isTuple)

    if dt == uint32 or dt == cuint:
        if left < 0: l = 0
        if right > uint32_max: dt = uint64
        else: return uint32_(l, r, n, p, seed, div, diff, isTuple)

    if dt == uint64 or dt == upointer:
        if left < 0: l = 0
        else: return uint64_(l, r, n, p, seed, div, diff, isTuple)

    ###
    if dt == int8:
        if right > int8_max: dt = int16
        else: return int8_(l, r, n, p, seed, div, diff, isTuple)

    if dt == int16:
        if right > int16_max: dt = int32
        else: return int16_(l, r, n, p, seed, div, diff, isTuple)

    if dt == int32 or dt == cint:
        if right > int32_max: dt = int64
        else: return int32_(l, r, n, p, seed, div, diff, isTuple)

    if dt == int64 or dt == pointer:
        return int64_(l, r, n, p, seed, div, diff, isTuple)

    ###
    if dt == float32:
        if right > float32_max: dt = float64
        else: return float32_(<float>left, <float>right, n, p, seed, div, diff, isTuple)

    return float64_(left, right, n, p, seed, div, diff, isTuple)


######    
cdef float Zmultf = 2.0 / <float>cycle
cdef double Zmultd = 2.0 / <double>cycle2


cdef void Z32_slow0(float[::1] out, UINT32 x, SIZE div, float mult1, float mult2) nogil:
    cdef SIZE i, j = 0
    # Tried checking unif_1*unif_1 >= 0.4, but it failed / too slow.
    cdef float unif_1, unif_2, normal = 2.0
    for i in range(div):
        while normal >= 1.0:
            x = (214013 * x + 2531011) & cycle;  unif_1 = <float>x*Zmultf - 1
            x = (214013 * x + 2531011) & cycle;  unif_2 = <float>x*Zmultf - 1
            normal = unif_1*unif_1 + unif_2*unif_2
        out[j] = unif_1 * mult1 * sqrt(-log(normal) / normal); j += 1
        out[j] = unif_2 * mult2; j += 1; normal = 2.0;
        
######
cdef void Z32_slow1(float[::1] out, UINT32 x, SIZE div, float mult1, float mult2, float mean) nogil:
    cdef SIZE i, j = 0
    cdef float unif_1, unif_2, normal = 2.0
    for i in range(div):
        while normal >= 1.0:
            x = (214013 * x + 2531011) & cycle;  unif_1 = <float>x*Zmultf - 1
            x = (214013 * x + 2531011) & cycle;  unif_2 = <float>x*Zmultf - 1
            normal = unif_1*unif_1 + unif_2*unif_2
        out[j] = unif_1 * mult1 * sqrt(-log(normal) / normal) + mean; j += 1
        out[j] = unif_2 * mult2 + mean; j += 1; normal = 2.0;
        
######
cdef void Z32_fast0(
    float[::1] out, UINT32 x, SIZE div, SIZE diff, SIZE n, float mult1, float mult2) nogil:
    cdef SIZE i, l, r
    l, r = 0, n
    cdef float unif_1, unif_2, temp1, temp2, normal = 2.0
    for i in range(div):
        while normal >= 1.0:
            x = (214013 * x + 2531011) & cycle;  unif_1 = <float>x*Zmultf - 1
            x = (214013 * x + 2531011) & cycle;  unif_2 = <float>x*Zmultf - 1
            normal = unif_1*unif_1 + unif_2*unif_2

        temp1 = unif_1*mult1*(1/(normal + 0.02) - 0.1);  temp2 = unif_2*mult2
        out[l] = temp1;      out[r] = temp2;         l += 1; r -= 1;     # reflection
        out[l] = temp2*.4;   out[r] = temp1*.4;      l += 1; r -= 1;
        out[l] = temp1*.75;  out[r] = temp2*.75;     l += 1; r -= 1; normal = 2.0;
        
    if diff >= 1:   out[l] = temp1*.2;   l += 1;
    if diff >= 2:   out[r] = -temp2*.2;  r -= 1;
    if diff >= 3:   out[l] = temp1*.1;   l += 1;
    if diff >= 4:   out[r] = -temp2*.1;  r -= 1;
    if diff >= 5:   out[l] = (temp1+temp2)/2

        
######
cdef void Z32_fast1(
    float[::1] out, UINT32 x, SIZE div, SIZE diff, SIZE n, float mult1, float mult2, float mean) nogil:
    cdef SIZE i, l, r
    l, r = 0, n
    cdef float unif_1, unif_2, temp1, temp2, normal = 2.
    for i in range(div):
        while normal >= 1.0:
            x = (214013 * x + 2531011) & cycle;  unif_1 = <float>x*Zmultf - 1
            x = (214013 * x + 2531011) & cycle;  unif_2 = <float>x*Zmultf - 1
            normal = unif_1*unif_1 + unif_2*unif_2

        temp1 = unif_1*mult1*(1/(normal + 0.02) - 0.1);      temp2 = unif_2*mult2
        out[l] = mean-temp1;      out[r] = mean-temp2;       l += 1; r -= 1;     # reflection
        out[l] = mean+temp2*.4;  out[r] = temp1*.4+mean;     l += 1; r -= 1;
        out[l] = mean-temp1*.75;  out[r] = temp2*.75+mean;   l += 1; r -= 1; normal = 2.0;
        
    if diff >= 1:   out[l] = temp1*.2+mean;   l += 1;
    if diff >= 2:   out[r] = mean-temp2*.2;   r -= 1;
    if diff >= 3:   out[l] = temp1*.1+mean;   l += 1;
    if diff >= 4:   out[r] = mean-temp2*.1;   r -= 1;
    if diff >= 5:   out[l] = (temp1+temp2)/2+mean
        

######
cdef void Z64_slow0(double[::1] out, UINT64 x, SIZE div, double mult1, double mult2) nogil:
    cdef SIZE i, j = 0
    cdef double unif_1, unif_2, normal = 2.0
    for i in range(div):
        while normal >= 1.0:
            x = (214013 * x + 2531011) & cycle2;  unif_1 = <double>x*Zmultd - 1
            x = (214013 * x + 2531011) & cycle2;  unif_2 = <double>x*Zmultd - 1
            normal = unif_1*unif_1 + unif_2*unif_2
        out[j] = unif_1 * mult1 * sqrt(-log(normal) / normal); j += 1
        out[j] = unif_2 * mult2; j += 1; normal = 2.0;
        
######
cdef void Z64_slow1(double[::1] out, UINT64 x, SIZE div, double mult1, double mult2, double mean) nogil:
    cdef SIZE i, j = 0
    cdef double unif_1, unif_2, normal = 2.0
    for i in range(div):
        while normal >= 1.0:
            x = (214013 * x + 2531011) & cycle2;  unif_1 = <double>x*Zmultd - 1
            x = (214013 * x + 2531011) & cycle2;  unif_2 = <double>x*Zmultd - 1
            normal = unif_1*unif_1 + unif_2*unif_2
        out[j] = unif_1 * mult1 * sqrt(-log(normal) / normal) + mean; j += 1
        out[j] = unif_2 * mult2 + mean; j += 1; normal = 2.0;
        
######
cdef void Z64_fast0(
    double[::1] out, UINT64 x, SIZE div, SIZE diff, SIZE n, double mult1, double mult2) nogil:
    cdef SIZE i, l, r
    l, r = 0, n
    cdef double unif_1, unif_2, temp1, temp2, normal = 2.0
    for i in range(div):
        while normal >= 1.0:
            x = (214013 * x + 2531011) & cycle2;  unif_1 = <double>x*Zmultd - 1
            x = (214013 * x + 2531011) & cycle2;  unif_2 = <double>x*Zmultd - 1
            normal = unif_1*unif_1 + unif_2*unif_2

        temp1 = unif_1*mult1*(1/(normal + 0.02) - 0.1);  temp2 = unif_2*mult2
        out[l] = -temp1;      out[r] = -temp2;           l += 1; r -= 1;     # reflection
        out[l] = temp2*.4;    out[r] = temp1*.4;         l += 1; r -= 1;
        out[l] = -temp1*.75;  out[r] = temp2*.75;        l += 1; r -= 1; normal = 2.0;
        
    if diff >= 1:   out[l] = temp1*.2;   l += 1;
    if diff >= 2:   out[r] = -temp2*.2;  r -= 1;
    if diff >= 3:   out[l] = temp1*.1;   l += 1;
    if diff >= 4:   out[r] = -temp2*.1;  r -= 1;
    if diff >= 5:   out[l] = (temp1+temp2)/2
        
######
cdef void Z64_fast1(
    double[::1] out, UINT64 x, SIZE div, SIZE diff, SIZE n, double mult1, double mult2, double mean) nogil:
    cdef SIZE i, l, r
    l, r = 0, n
    cdef double unif_1, unif_2, temp1, temp2, normal = 2.
    for i in range(div):
        while normal >= 1.0:
            x = (214013 * x + 2531011) & cycle2;  unif_1 = <double>x*Zmultd - 1
            x = (214013 * x + 2531011) & cycle2;  unif_2 = <double>x*Zmultd - 1
            normal = unif_1*unif_1 + unif_2*unif_2

        temp1 = unif_1*mult1*(1/(normal + 0.02) - 0.1);      temp2 = unif_2*mult2
        out[l] = mean-temp1;      out[r] = temp2-mean;       l += 1; r -= 1;     # reflection
        out[l] = mean+temp2*.4;   out[r] = temp1*.4+mean;    l += 1; r -= 1;
        out[l] = mean-temp1*.75;  out[r] = temp2*.75+mean;   l += 1; r -= 1; normal = 2.0;
        
    if diff >= 1:   out[l] = temp1*.2+mean;   l += 1;
    if diff >= 2:   out[r] = mean-temp2*.2;   r -= 1;
    if diff >= 3:   out[l] = temp1*.1+mean;   l += 1;
    if diff >= 4:   out[r] = mean-temp2*.1;   r -= 1;
    if diff >= 5:   out[l] = (temp1+temp2)/2+mean

        
######
cdef double sqrt2 = <double> sqrt(2.0)
cdef float sqrt2f = <float> sqrt(2.0)
######
cdef ARRAY Z64_(SIZE n, SIZE p, int seed, SIZE div, bool isTuple, bool isSlow,
    double mean, double std, bool noMean):

    cdef double[::1] out
    cdef double[:,::1] out2D
    cdef SIZE i, j, diff
    cdef double mult1 = sqrt2*std
    cdef double mult2 = <double>(2.0 if isSlow else 2.3)*std
    cdef UINT64 x = <UINT64> (rand() if seed < 0 else seed)
    cdef UINT64 change = <UINT64> (214013 * x + 2531011)

    if isTuple:
        out2D = np.zeros((n,p), dtype = np.float64)
        if not isSlow: div, diff, p = p//6, p%6, p-1
        with nogil:
            if noMean:
                if isSlow: # slow but accurate.
                    for i in range(n): 
                        Z64_slow0(out2D[i], change+<UINT64>i, div, mult1, mult2)
                else: # fast but less accurate
                    for i in range(n): 
                        Z64_fast0(out2D[i], change+<UINT64>i, div, diff, p, mult1, mult2)
            else:
                if isSlow:
                    for i in range(n): 
                        Z64_slow1(out2D[i], change+<UINT64>i, div, mult1, mult2, mean)
                    if p%2 == 1:
                        j = p - 1
                        for i in range(n): out2D[i, j] = mean
                else:
                    for i in range(n): 
                        Z64_fast1(out2D[i], change+<UINT64>i, div, diff, p, mult1, mult2, mean)
    else:
        out = np.zeros(n, dtype = np.float64)
        if not isSlow: div, diff, n = n//6, n%6, n-1
        if noMean:
            if isSlow:      Z64_slow0(out, x, div, mult1, mult2)
            else:           Z64_fast0(out, x, div, diff, n, mult1, mult2)
        else:
            if isSlow:      
                Z64_slow1(out, x, div, mult1, mult2, mean)
                if n%2 == 1:    out[n-1] = mean
            else:
                Z64_fast1(out, x, div, diff, n, mult1, mult2, mean)
    return np.asarray(out2D) if isTuple else np.asarray(out)

######
cdef ARRAY Z32_(SIZE n, SIZE p, int seed, SIZE div, bool isTuple, bool isSlow,
    float mean, float std, bool noMean):

    cdef float[::1] out
    cdef float[:,::1] out2D
    cdef SIZE i, j, diff
    cdef float mult1 = sqrt2*std
    cdef float mult2 = <float>(2.0 if isSlow else 2.3)*std
    cdef UINT32 x = <UINT32> (rand() if seed < 0 else seed)
    cdef UINT32 change = <UINT32> (214013 * x + 2531011)

    if isTuple:
        out2D = np.zeros((n,p), dtype = np.float32)
        if not isSlow: div, diff, p = p//6, p%6, p-1
        with nogil:
            if noMean:
                if isSlow: # slow but accurate.
                    for i in range(n): 
                        Z32_slow0(out2D[i], change+<UINT32>i, div, mult1, mult2)
                else: # fast but less accurate
                    for i in range(n): 
                        Z32_fast0(out2D[i], change+<UINT32>i, div, diff, p, mult1, mult2)
            else:
                if isSlow:
                    for i in range(n): 
                        Z32_slow1(out2D[i], change+<UINT32>i, div, mult1, mult2, mean)
                    if p%2 == 1:
                        j = p - 1
                        for i in range(n): out2D[i, j] = mean
                else:
                    for i in range(n): 
                        Z32_fast1(out2D[i], change+<UINT32>i, div, diff, p, mult1, mult2, mean)
    else:
        out = np.zeros(n, dtype = np.float32)
        if not isSlow: div, diff, n = n//6, n%6, n-1
        if noMean:
            if isSlow:      Z32_slow0(out, x, div, mult1, mult2)
            else:           Z32_fast0(out, x, div, diff, n, mult1, mult2)
        else:
            if isSlow:      
                Z32_slow1(out, x, div, mult1, mult2, mean)
                if n%2 == 1:    out[n-1] = mean
            else:
                Z32_fast1(out, x, div, diff, n, mult1, mult2, mean)
    return np.asarray(out2D) if isTuple else np.asarray(out)


######
cpdef ARRAY normal(double mean = 0, double std = 1, size = 10, 
    int seed = -1, dtype = np.float32, str mode = "fast"):
    """
    Creates normal random numbers. Uses ideas from the Box Muller Transfrom,
    and inspired from (senderle)'s Stackoverflow implementation of fast
    normal numbers. Also uses a modified Linear Congruential Generator for
    uniform numbers. There are also 2 modes (slow, fast). Slow is more accurate.
    Fast approximates sqrt(-2log(x)/x) as sqrt(2)(1/(x+0.03)-0.8)
    """
    cdef char dt
    cdef SIZE div, n, p, diff
    cdef bool isTuple, noMean = (mean == 0)
    div, n, p, diff, isTuple, dt = args_process(dtype, size, 2)
    cdef bool isSlow = (mode.lower() == "slow")

    ###
    if dt == float64:
        return Z64_(n, p, seed, div, isTuple, isSlow, mean, std, noMean)

    return Z32_(n, p, seed, div, isTuple, isSlow, <float>mean, <float>std, noMean)
