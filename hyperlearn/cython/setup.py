

from distutils.core import setup
from Cython.Build import cythonize
from numpy import get_include
from multiprocessing import cpu_count
from Cython.Compiler import Options

Options.docstrings = False
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
    ),
    include_dirs = [get_include()]
)
