/*
	Headers is the main header file for the entire C, C++ layer.

	Author:  Copyright 2022-, Daniel Han-Chen, Hyperlearn, Moonshot
	License: Apache 2 License
*/

#pragma once
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <float.h>
#include <string.h>
#include <stdexcept>
#include <limits.h>

// Using Hedley, https://nemequ.github.io/hedley/api-reference.html,
// we can carefully determine the compiler.

#define MIN(a, b)			((a) < (b) ? (a) : (b))
#define MAX(a, b)			((a) > (b) ? (a) : (b))
#define CEILDIV(a, b)		(((a) / (b)) + (((a) % (b))!=0))
#define CEIL_DIV(a, b)		CEILDIV(a, b)

// Force inline macros
#define __INLINE__ 				always_inline
// Place __FENCE__ after intrinsic functions so to force the compiler
// to not re-order the instruction.
#define __FENCE__				__asm__ __volatile__ ("":::"memory")
// Some macros need nothing, so define it
#define __NOTHING__

// std::pair is slightly faster
#define MULTIPLE_RETURN2(...) 	std::pair<__VA_ARGS__>
#define MULTIPLE_RETURN(...) 	std::tuple<__VA_ARGS__>
#define RETURN_MULTIPLE2(...) 	std::make_pair(__VA_ARGS__)
#define RETURN_MULTIPLE(...) 	std::make_tuple(__VA_ARGS__)
#define UNPACK_MULTIPLE(...) 	std::tie(__VA_ARGS__)
#define UNPACK_MULTIPLE2(...) 	std::tie(__VA_ARGS__)
#define SAME_TYPE(a, b)     	(std::is_same<a, b>::value)
// determines type of object, with the removal of the reference ie &x0 --> x0
#define GET_TYPE(a)				std::remove_reference_t<decltype(a)>
#define SWAP(a, b)				{GET_TYPE(a) __TEMP__ = a; a = b; b = __TEMP__;}

// UNLIKELY_ON enables other strange dtypes (uint16 etc)
#define UNLIKELY_ON         	(true)
#define _ __restrict
#define LIKELY(x)           	(__builtin_expect(!!(x), 1))
#define UNLIKELY(x)         	(__builtin_expect(!!(x), 0))


#if defined(__clang__)
	#warning "Clang is NOT supported for Assembly optimizations! Using generic optimizations"
	#ifdef __HAS_SIMD__
		#undef __HAS_SIMD__
		#define __HAS_SIMD__ 	(false)
	#endif

	#define HALF_LIKELY(x)  	(__builtin_unpredictable((x)))
	#define HALF_UNLIKELY		HALF_LIKELY
	#define VERY_UNLIKELY(x) 	(__builtin_expect(!!(x), 0))

#elif defined(__GNUC__) || defined(__GNUG__) || defined(__MINGW32__) || defined(__MINGW64__)

	#if (__GNUC__ >= 9)
		#define HALF_LIKELY(x)  	(__builtin_expect_with_probability(!!(x), 1, 0.5f))
		#define HALF_UNLIKELY		HALF_LIKELY
		#define VERY_UNLIKELY(x) 	(__builtin_expect_with_probability(!!(x), 0, 0.01f))
	#else
		#define HALF_LIKELY(x)  	(x)
		#define HALF_UNLIKELY		HALF_LIKELY
		#define VERY_UNLIKELY(x) 	(__builtin_expect(!!(x), 0))
	#endif

#elif defined(_WIN64) || defined(_WIN32) || defined(_MSC_VER)
	#error MSVC compiler is not supported. Use either Clang or GCC.

#else
	#error Compiler is not supported. Use either Clang or GCC.
#endif


// Unrolling macros
#if defined(__clang__)
	// Currently clang Assembly optimizations is NOT supported
	#define __UNROLL0__ _Pragma("clang loop unroll(disable)")
	#define __UNROLL1__ _Pragma("clang loop unroll_count(1)")
	#define __UNROLL2__ _Pragma("clang loop unroll_count(2)")
	#define __UNROLL4__ _Pragma("clang loop unroll_count(4)")
	#define __UNROLL8__ _Pragma("clang loop unroll_count(8)")
	#define __O1__
	#define __O2__
	#define __ALIGN_LOOPS__
	#define __UNROLL_LOOPS__ 
	#define __NO_BUILTIN__
	#define __COLD__ cold

#elif defined(__GNUC__) || defined(__GNUG__) || defined(__MINGW32__) || defined(__MINGW64__)
	#define __UNROLL0__ _Pragma("GCC unroll 0")
	#define __UNROLL1__ _Pragma("GCC unroll 1")
	#define __UNROLL2__ _Pragma("GCC unroll 2")
	#define __UNROLL4__ _Pragma("GCC unroll 4")
	#define __UNROLL8__ _Pragma("GCC unroll 8")
	// Unroll and jam causes excessive unrolling
	// Peel loops cause unrolling of small loops.
	// Sadly GCC -O3 then adding fno-... doesnt work!!!
	// So use -O2 and just extend it

	/*"loop-unroll-and-jam", \*/
	/*"peel-loops", \*/
	/*"version-loops-for-strides"*/ // This fails on ARM compilers
	#define __O3_ARGS__ \
		"gcse-after-reload", \
		"ipa-cp-clone", \
		"loop-interchange", \
		"predictive-commoning", \
		"split-loops", \
		"split-paths", \
		"tree-loop-distribution", \
		"tree-loop-vectorize", \
		"tree-partial-pre", \
		"tree-slp-vectorize", \
		"unswitch-loops", \
		"vect-cost-model", \
		"vect-cost-model=dynamic"

	#define __O2__ optimize("O2", "fast-math", "align-functions=16", __O3_ARGS__)
	#define __O1__ optimize("O2", "fast-math", "align-functions=16")
	
	#if (__ENABLE_AVX512__)
		#define __ALIGN_LOOPS__ optimize("O2", "fast-math", "align-loops=64", "align-functions=16", __O3_ARGS__)
		#define __UNROLL_LOOPS__ optimize("O3", "fast-math", "unroll-loops", "align-loops=64", "align-functions=16")
		#define __NO_BUILTIN__ optimize("O2", "fast-math", "no-tree-loop-distribute-patterns", "align-loops=64", "align-functions=16", __O3_ARGS__)
	#elif (__ENABLE_AVX__)
		#define __ALIGN_LOOPS__ optimize("O2", "fast-math", "align-loops=32", "align-functions=16", __O3_ARGS__)
		#define __UNROLL_LOOPS__ optimize("O3", "fast-math", "unroll-loops", "align-loops=32", "align-functions=16")
		#define __NO_BUILTIN__ optimize("O2", "fast-math", "no-tree-loop-distribute-patterns", "align-loops=32", "align-functions=16", __O3_ARGS__)
	#else
		#define __ALIGN_LOOPS__ optimize("O2", "fast-math", "align-loops=16", "align-functions=16", __O3_ARGS__)
		#define __UNROLL_LOOPS__ optimize("O3", "fast-math", "unroll-loops", "align-loops=16", "align-functions=16")
		#define __NO_BUILTIN__ optimize("O2", "fast-math", "no-tree-loop-distribute-patterns", "align-loops=16", "align-functions=16", __O3_ARGS__)
	#endif

	#define __COLD__ cold

#endif


// Cachesize determination and functions
#include "./determine_cachesize.h"
// End cachesize determination


// Safe string copy
#define strncpy(dest, src, len) \
{ \
	strncpy(dest, src, len-1);  \
	dest[len-1] = '\0'; \
}


#define printf(...) fprintf(stderr, __VA_ARGS__)

#define NO_MEMORY() fprintf(stderr, "No memory line[%d] file[%s]\n", __LINE__, __FILE__)


#if (__has_include(<Python.h>))
	#include <Python.h>

	__attribute__((__O1__, noclone, nonnull, cold, noinline))
	void
	__Pyx_CppExn2PyErr__(void)
	{
		try {
			if (PyErr_Occurred())
				; // let the latest Python exn pass through and ignore the current one
			else
				throw;
		}
		catch (const std::invalid_argument& exn) {
			PyErr_SetString(PyExc_TypeError, exn.what());
		}
		catch (const std::bad_alloc& exn) {
			PyErr_SetString(PyExc_MemoryError, exn.what());
		} /* catch (const std::bad_cast& exn) {
		PyErr_SetString(PyExc_TypeError, exn.what());
		} catch (const std::bad_typeid& exn) {
		PyErr_SetString(PyExc_TypeError, exn.what());
		} catch (const std::domain_error& exn) {
		PyErr_SetString(PyExc_ValueError, exn.what());
		} catch (const std::ios_base::failure& exn) {
		PyErr_SetString(PyExc_IOError, exn.what());
		} catch (const std::out_of_range& exn) {
		PyErr_SetString(PyExc_IndexError, exn.what());
		} catch (const std::overflow_error& exn) {
		PyErr_SetString(PyExc_OverflowError, exn.what());
		} catch (const std::range_error& exn) {
		PyErr_SetString(PyExc_ArithmeticError, exn.what());
		} catch (const std::underflow_error& exn) {
		PyErr_SetString(PyExc_ArithmeticError, exn.what());
		} catch (const std::exception& exn) {
		PyErr_SetString(PyExc_RuntimeError, exn.what());
		}
		*/
		catch (...) {
			PyErr_SetString(PyExc_RuntimeError, "Exception");
		}
	}
	#define __Pyx_CppExn2PyErr __Pyx_CppExn2PyErr__
#endif

__attribute__((__O1__, noclone, nonnull, cold, __INLINE__))
static void inline
_raise_error(const char *_ x)
{
	throw std::invalid_argument(x);
}

__attribute__((__O1__, noclone, nonnull, cold, __INLINE__))
static void inline
_raise_internal_error(void)
{
	throw;
}

__attribute__((__O1__, noclone, nonnull, cold, __INLINE__))
static void inline
_raise_no_memory(const char *_ x)
{
	fprintf(stderr, "%s", x);
	throw std::bad_alloc();
}

#define raise_error(x) 				_raise_error(x)
#define raise_internal_error(x) 	_raise_internal_error(x)
#define raise_no_memory(x) 			_raise_no_memory(x)


/*
	Include SSE, AVX, AVX2, AVX512 intrinsics.
*/
#include "./intrinsics/intrinsics.h"

// OpenMP SIMD vectorization lengths. Only works in GCC 10
#define __DO_PRAGMA__(x) 	_Pragma (#x)
#define __DO_PRAGMA(x) 		__DO_PRAGMA__(x)

#if (!defined(__NFLOATS__))
	#define __NFLOATS__				(16)
	#define __NINTS__ 				(16)
	#define __ALIGNMENT__			(64)
	#define __OMP_SIMD_AVX512__(x) 	__DO_PRAGMA(omp simd x)
	#define __OMP_SIMD_AVX__(x)		__DO_PRAGMA(omp simd x)
	#define __OMP_SIMD__(x) 		__DO_PRAGMA(omp simd x)
#else
	#define __OMP_SIMD_AVX512__(x) 	__DO_PRAGMA(omp simd simdlen(__NFLOATS__) x)
	#define __OMP_SIMD_AVX__(x)		__DO_PRAGMA(omp simd x)
	#define __OMP_SIMD__(x) 		__DO_PRAGMA(omp simd x)
#endif

/*
	End Include SSE, AVX, AVX2, AVX512 intrinsics.
*/


__attribute__((__O2__, hot, __INLINE__))
inline size_t
ITEMSIZE(const char dtype)
{
	switch (dtype)
	{
		case 'f': return 4;
		case 'd': return 8;
		
		case 'l': return 4;
		case 'q': return 8;
		case 'L': return 4;
		case '?':
		case 'B': return 1;
		case 'H': return 2;
		
		case 'Q': return 8;
		case 'b': return 1;
		case 'h': return 2;

		default: __builtin_unreachable();
	}
	__builtin_unreachable();
}

// Strangely GCC doesn't recognise we are dividing by a multiple of 2.
// So, force GCC to optimize better via a switch statement.
__attribute__((__O2__, hot, __INLINE__))
inline size_t
__FIND_STRIDES__(size_t size,
				 const char dtype)
{
	switch (dtype)
	{
		case 'f': return size / 4;
		case 'd': return size / 8;
		
		case 'l': return size / 4;
		case 'q': return size / 8;
		case 'L': return size / 4;
		case '?':
		case 'B': return size / 1;
		case 'H': return size / 2;
		
		case 'Q': return size / 8;
		case 'b': return size / 1;
		case 'h': return size / 2;

		default: __builtin_unreachable();
	}
	__builtin_unreachable();
}
#define FIND_STRIDES(strides, dtype) strides = __FIND_STRIDES__(strides, dtype)


__attribute__((__INLINE__)) inline char DTYPE(const float *_ a)    { return 'f'; }
__attribute__((__INLINE__)) inline char DTYPE(const double *_ a)   { return 'd'; }
__attribute__((__INLINE__)) inline char DTYPE(const uint8_t *_ a)  { return 'B'; }
__attribute__((__INLINE__)) inline char DTYPE(const uint16_t *_ a) { return 'H'; }
__attribute__((__INLINE__)) inline char DTYPE(const uint32_t *_ a) { return 'L'; }
__attribute__((__INLINE__)) inline char DTYPE(const int32_t *_ a)  { return 'l'; }
__attribute__((__INLINE__)) inline char DTYPE(const int64_t *_ a)  { return 'q'; }
__attribute__((__INLINE__)) inline char DTYPE(const int8_t *_ a)   { return 'b'; }
__attribute__((__INLINE__)) inline char DTYPE(const int16_t *_ a)  { return 'h'; }
__attribute__((__INLINE__)) inline char DTYPE(const uint64_t *_ a) { return 'Q'; }


#if (UNLIKELY_ON == true)
	#define MULTI_FUNCTION_DISPATCH(fx) \
		switch (dtype) { \
			case 'f': CALL_FUNCTION(fx, float); 	break; \
			case 'd': CALL_FUNCTION(fx, double); 	break; \
			\
			case 'l': CALL_FUNCTION(fx, int32_t); 	break; \
			case 'q': CALL_FUNCTION(fx, int64_t); 	break; \
			case 'L': CALL_FUNCTION(fx, uint32_t); 	break; \
			case '?': \
			case 'B': CALL_FUNCTION(fx, uint8_t); 	break; \
			case 'H': CALL_FUNCTION(fx, uint16_t); 	break; \
			\
			case 'Q': CALL_FUNCTION(fx, uint64_t); 	break; \
			case 'b': CALL_FUNCTION(fx, int8_t); 	break; \
			case 'h': CALL_FUNCTION(fx, int16_t); 	break; \
			\
			default: { \
				raise_error("Data type is not recognised at all???"); \
			} \
		}

	#define FAST_FUNCTION_DISPATCH(fx) \
		MULTI_FUNCTION_DISPATCH(fx)

#else
	#define MULTI_FUNCTION_DISPATCH(fx) \
		switch (dtype) { \
			case 'f': CALL_FUNCTION(fx, float); 	break; \
			case 'd': CALL_FUNCTION(fx, double); 	break; \
			\
			case 'L': CALL_FUNCTION(fx, uint32_t); 	break; \
			case 'l': CALL_FUNCTION(fx, int32_t); 	break; \
			case 'q': CALL_FUNCTION(fx, int64_t); 	break; \
			case '?': \
			case 'B': CALL_FUNCTION(fx, uint8_t); 	break; \
			case 'H': CALL_FUNCTION(fx, uint16_t); 	break; \
			\
			default: { \
				raise_error("Data type is not recognised? Change UNLIKELY_ON to true."); \
			} \
		}

	#define FAST_FUNCTION_DISPATCH(fx) \
		if LIKELY(dtype == 'f') { \
			CALL_FUNCTION(fx, float); \
		} \
		else if UNLIKELY(dtype == 'd') { \
			CALL_FUNCTION(fx, double); \
		} \
		else { \
			raise_error("Data type is not recognised? Change UNLIKELY_ON to true."); \
		}
#endif


// Alignment and unused macros
#ifndef __ALIGNMENT__
	#define __ALIGNMENT__ (16)
#endif
#define ALIGNED             __attribute__((aligned(__ALIGNMENT__)))
#define ASSUME_ALIGNED(x)   (__builtin_assume_aligned((x), __ALIGNMENT__))
#define PACKED              __attribute__((packed))
#define UNUSED              __attribute__((unused))


// Aligned malloc / free
// Will throw exception if failed
#define aligned_malloc(size) 	operator new[](size, std::align_val_t{__ALIGNMENT__})
#define aligned_free(ptr)		operator delete[](ptr, std::align_val_t{__ALIGNMENT__})


// To counteract false sharing [2 CPUs access same cache line], use attribute TRUE_SHARING
// https://en.cppreference.com/w/cpp/thread/hardware_destructive_interference_size
// According to https://www.youtube.com/watch?v=h58X-PaEGng [Episode 5.9 - Elimination of False Cache Line Sharing]
// 256 bytes must be used on new Intel processors.
#ifdef __cpp_lib_hardware_interference_size
	constexpr static const size_t __false_sharing_size__ = \
		MAX(std::hardware_destructive_interference_size, 256);
	#define TRUE_SHARING 	alignas(__false_sharing_size__)
#else
	// Lucky guess │ __cacheline_aligned │ L1_CACHE_BYTES │ L1_CACHE_SHIFT │ ...
	constexpr static const size_t __false_sharing_size__ = \
		MAX((2 * sizeof(std::max_align_t)), 256);
	#define TRUE_SHARING 	alignas(__false_sharing_size__)
#endif


// Not implemented functions
#define NOT_IMPLEMENTED(function) \
	__attribute__((__O2__, __INLINE__)) \
	inline auto \
	function { \
		throw std::runtime_error("Function not Implemented!"); \
	} \


/*
	Array and other Python wrappers
*/
#include "./python_object.h"
/*
	End Array and other Python wrappers
*/

/*
	Numpy and datatype definitions
*/
struct strides_t
{
	size_t strides[2];
};

// https://github.com/lattera/glibc/blob/master/sysdeps/ieee754/ieee754.h
// https://stackoverflow.com/questions/15685181/how-to-get-the-sign-mantissa-and-exponent-of-a-floating-point-number
union bfloat16_t
{
	int16_t bh;
	int16_t i;
	uint16_t u;

	struct
	{
	#if (__BYTE_ORDER__ == __ORDER_BIG_ENDIAN__)
		int16_t sign 		: 1;
		int16_t exponent 	: 8;
		int16_t mantissa 	: 7;
	#else
		// x86 is Little Endian
		int16_t mantissa 	: 7;
		int16_t exponent 	: 8;
		int16_t sign 		: 1;
	#endif
	};
};
typedef int16_t bfloat_t;

union bfloat32_t
{
	float f;
	int32_t i;
	uint32_t u;

	struct
	{
	#if (__BYTE_ORDER__ == __ORDER_BIG_ENDIAN__)
		int16_t bh;
		int16_t mantissa2;
	#else
		// x86 is Little Endian
		int16_t mantissa2;
		int16_t bh;
	#endif
	};
};


__attribute__((__O2__, hot, __INLINE__))
inline bfloat_t
FLOAT_TO_BFLOAT(float x)
{
	// https://stackoverflow.com/questions/55253233/convert-fp32-to-bfloat16-in-c
	x = x * 1.001957f;
	bfloat32_t f;
	f.f = x;
	return f.bh;
}

__attribute__((__O2__, hot, __INLINE__))
inline bfloat16_t
FLOAT_TO_BFLOAT(bfloat32_t x)
{
	// https://stackoverflow.com/questions/55253233/convert-fp32-to-bfloat16-in-c
	x.f = x.f * 1.001957f;
	bfloat32_t f;
	f.f = x.f;
	bfloat16_t bh;
	bh.bh = f.bh;
	return bh;
}

__attribute__((__O2__, hot, __INLINE__))
inline float
BFLOAT_TO_FLOAT(bfloat_t x)
{
	bfloat32_t f;
	f.bh = x;
	f.mantissa2 = 0;
	return f.f;
}

__attribute__((__O2__, hot, __INLINE__))
inline bfloat32_t
BFLOAT_TO_FLOAT(bfloat16_t x)
{
	bfloat32_t f;
	f.bh = x.bh;
	f.mantissa2 = 0;
	return f;
}

__attribute__((__O2__, hot, __INLINE__))
inline bfloat_t
bhabs(bfloat_t x)
{
	return x & 0b0'1111'1111'1111'111;
}
/*
	End Numpy and datatype definitions
*/
