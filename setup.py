
from setuptools import setup

install_requires = [
        'numpy >= 1.13.0',
        'torchvision >= 0.2.0',
        'scikit-learn >= 0.18.0',
        'scipy >= 1.0.0',
        'pandas >= 0.21.0',
        'torch >= 0.4.0',
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


setup(name = 'hyperlearn',
      version = '0.0.1',
      author = 'Daniel Han-Chen & Others listed on Github',
      url = 'https://github.com/danielhanchen/hyperlearn',
      long_description = desc,
      py_modules = ['hyperlearn'],
      install_requires = install_requires,
      dependency_links = dependency_links,
      classifiers=[  # Optional
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
    		]
      )
