
# python setup.py build_ext --inplace
from distutils.core import setup
from Cython.Build import cythonize
from numpy import get_include
from Cython.Compiler import Options
import os
os.environ['CFLAGS'] = '-O3 -march=native'
os.environ['CXXFLAGS'] = '-O3 -march=native'
os.environ['CL'] = '/arch:AVX /arch:AVX2 /arch:SSE2 /arch:SSE /arch:ARMv7VE /arch:VFPv4'

Options.docstrings = True
Options.generate_cleanup_code = True

setup(
    ext_modules = cythonize("*.pyx",
        compiler_directives = {
            'language_level':3, 
            'boundscheck':False, 
            'wraparound':False,
            'initializedcheck':False, 
            'cdivision':True,
            'nonecheck':False,
        },
        quiet = True,
        force = True,
        build_dir = "C_files",
    ),
    include_dirs = [get_include()],
)
