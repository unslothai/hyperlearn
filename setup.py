

from distutils.core import setup
from setuptools.command.install import install
from setuptools import find_packages
import subprocess

from Cython.Build import cythonize
from numpy import get_include
from Cython.Compiler import Options
import os

os.environ['CFLAGS'] = '-O3 -march=native -ffast-math -mtune=native -ftree-vectorize'
os.environ['CXXFLAGS'] = '-O3 -march=native -ffast-math -mtune=native -ftree-vectorize'
os.environ['CL'] = '/arch:AVX /arch:AVX2 /arch:SSE2 /arch:SSE /arch:ARMv7VE /arch:VFPv4'

Options.docstrings = True
Options.generate_cleanup_code = True

install_requires = [
    'numpy >= 1.13.0',
    'torchvision >= 0.2.0',
    'scikit-learn >= 0.18.0',
    'scipy >= 1.0.0',
    'pandas >= 0.21.0',
    'torch >= 0.4.0',
    'numba >= 0.37.0',
    'psutil >= 4.0.0',
    "cython >= 0.x",
    ]

dependency_links = [
    ]


desc = """\
HyperLearn

Faster, Leaner Scikit Learn (Sklearn) morphed with Statsmodels & 
Deep Learning drop in substitute. Designed for big data, HyperLearn 
can use 50%+ less memory, and runs 50%+ faster on some modules. 
Will have GPU support, and all modules are parallelized. 
Written completely in PyTorch, Numba, Numpy, Pandas, Scipy & LAPACK.

https://github.com/danielhanchen/hyperlearn
"""


# class InstallLocalPackage(install):
#     def run(self):
#         install.run(self)
#         print("******* Now compiling C and Cython code..... *******")

#         subprocess.call(
#             "python setup.py build_ext --inplace", shell = True,
#             cwd = "hyperlearn/cython"
#         )


# https://github.com/mozilla/treeherder/commit/f17bcf82051300ce1ff012dc7f1d42919137800a
if os.environ.get('READTHEDOCS'):
    ext_modules = []
else:
    ext_modules = cythonize("hyperlearn/cython/*.pyx",
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
    )



## Contributed by themightyoarfish [6/1/19 Issue 13]
kwargs = {
    "name" : 'hyperlearn',
    "version" : '0.0.1',
    "author" : 'Daniel Han-Chen & Others listed on Github',
    "url" : 'https://github.com/danielhanchen/hyperlearn',
    "long_description" : desc,
    "py_modules" : ['hyperlearn'],
    'packages' : find_packages(""),
    "install_requires" : install_requires,
    "dependency_links" : dependency_links,
    "classifiers" : [  # Optional
    'Development Status :: 1 - Planning',

    # Indicate who your project is intended for
    'Intended Audience :: Developers',
    'Intended Audience :: Education',
    'Intended Audience :: Science/Research',

    # Pick your license as you wish
    'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',

    # Specify the Python versions you support here. In particular, ensure
    # that you indicate whether you support Python 2, Python 3 or both.
    'Programming Language :: Python :: 3.4',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',

    'Topic :: Scientific/Engineering',
    'Topic :: Scientific/Engineering :: Mathematics',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'Topic :: Software Development',
    'Topic :: Software Development :: Libraries',
    'Topic :: Software Development :: Libraries :: Python Modules',
        ],
    #"cmdclass" : { 'install': InstallLocalPackage },
    "ext_modules" : ext_modules,
    "include_dirs" : [get_include()],
}

print("#### Welcome to Umbra's HyperLearn! ####")
print("#### During installation, code will be compiled down to C / LLVM via Numba. ####")
print("#### This could mean you have to wait...... ####")
print("\n#### You MUST have a C compiler AND MKL/LAPACK enabled Scipy. ####")
print("#### If you have Anaconda, then you are set to go! ####")


setup(**kwargs)

print("#### HyperLearn has been installed! ####")
print("\n#### If you want to compile Numba code, please run:")
print("    >>>>   python numba_compile.py")