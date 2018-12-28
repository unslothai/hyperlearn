
# python setup.py build_ext --inplace
from distutils.core import setup
from Cython.Build import cythonize
import numpy as np

setup(
    ext_modules=cythonize("base.pyx",
    	 compiler_directives = {
    	 'language_level':3, 
    	 'boundscheck':False, 
    	 'wraparound':False,
    	 'initializedcheck':False, 
    	 'cdivision':True,
    	 'nonecheck':False
    	 }),
    include_dirs=[np.get_include()],
)
