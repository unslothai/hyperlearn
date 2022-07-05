"""
	Prologue sets up the interaction layer between Cython, C and Python.
	We find the # of CPUs, define some commonly used variables, and allow
	the C level code to access those variables.

	Author:  Copyright 2022-, Daniel Han-Chen, Hyperlearn, Moonshot
	License: Apache 2 License
"""

cdef extern from * nogil:
	"""
	extern inline size_t MAX_THREADS;
	extern inline int PROC_BIND;
	#if (__has_include(<omp.h>))
		#include <omp.h>
		#define __HAS_OMP__ (true)
	#else
		#define __HAS_OMP__ (false)
		#define omp_get_num_procs() 1
		#define omp_get_proc_bind() 0
	#endif
	"""
	size_t MAX_THREADS
	size_t ALL_THREADS
	int    HAS_HYPERTHREADING
	const int __HAS_OMP__
	int PROC_BIND
	size_t omp_get_num_procs()
	int omp_get_proc_bind()
pass


# Get # of CPUs. ALL_THREADS includes hyperthreading.
# MAX_THREADS removes hyperthreading.
import psutil
ALL_THREADS = psutil.cpu_count(logical = True)
MAX_THREADS = psutil.cpu_count(logical = False)
if MAX_THREADS == 0:
	MAX_THREADS = 1
	ALL_THREADS = 1
pass
if MAX_THREADS != ALL_THREADS:
	HAS_HYPERTHREADING = 1
else:
	HAS_HYPERTHREADING = 0
pass


"""
	Process binding methods
	https://www.openmp.org/spec-html/5.0/openmpsu132.html#x169-7730003.2.23

	typedef enum omp_proc_bind_t { 
		omp_proc_bind_false = 0, 
		omp_proc_bind_true = 1, 
		omp_proc_bind_master = 2, 
		omp_proc_bind_close = 3, 
		omp_proc_bind_spread = 4 
	} omp_proc_bind_t;
"""
if (__HAS_OMP__ == 0):
	PROC_BIND = 0
else:
	PROC_BIND = <int>omp_get_proc_bind()
pass


cdef extern from * nogil:
	"""
	size_t MAX_THREADS;
	size_t ALL_THREADS;
	int PROC_BIND;
	"""
pass


# Other general type defines
ctypedef bint bool
DEF F = "fortran"
DEF C = "c"
DEF FLOAT32 = b"f"
DEF UINT8 = b"B"
DEF FLOAT64 = b"d"
DEF INT32 = b"l"
pass


# Library type imports
from libc.stdint cimport uint8_t, uint16_t, uint32_t, uint64_t
from libc.stdint cimport  int8_t,  int16_t,  int32_t,  int64_t
from libc.stdlib cimport malloc, free, realloc
from cpython.ref cimport PyTypeObject, PyObject, Py_INCREF, Py_DECREF
from cpython.ref cimport PyObject
pass


# Math library imports
cdef extern from "<math.h>" nogil:
	float fabs  (float x)
	float fminf (float x, float y)
	float fmaxf (float x, float y)
	float log10f(float x)
	float logf  (float x)
	float log2f (float x)
	float roundf(float x)
	float sqrtf (float x)
	float powf  (float x, float exponent)
	float ceil  (float x)
pass


# String library imports
cdef extern from "<string.h>" nogil:
	void memcpy(void *dest, void *src, size_t x)
	void memset(void *dest, int c,     size_t x)
	int  memcmp(void *dest, void *src, size_t x)
pass


# Char manipulations
cdef extern from "<ctype.h>" nogil:
	char tolower(char x)
	char toupper(char x)
pass


"""
Header file imports.
"""
cdef extern from "./headers.h" nogil:
	bool IS_ITERABLE(const PyObject *x)
	size_t ITERABLE_LENGTH(const PyObject *x)
	void *aligned_malloc(const size_t x)
	void *aligned_free(void *x)
	bool UNLIKELY(const bool cond)
	bool VERY_UNLIKELY(const bool cond)
	bool HALF_LIKELY(const bool cond)
	bool HALF_UNLIKELY(const bool cond)
	bool LIKELY(const bool cond)
	float FLT_EPSILON
	float FLT_MAX
	int MIN(int a, int b)
	int MAX(int a, int b)
	float MIN(float a, float b)
	float MAX(float a, float b)
	void raise_error(const char *x) except+
	void raise_no_memory(const char *x) except+
	const int __ALIGNMENT__
	void printf(const char *template, ...)
	size_t ITEMSIZE(const char dtype)

	void OBJECT_TO_ARRAY(PyObject *obj,
						 void *out,
						 const Py_ssize_t max_length,
						 const char *dtype) except+
