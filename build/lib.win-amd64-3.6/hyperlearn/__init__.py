
import os
if not os.path.isdir("./numba/__pycache__"):
	print("Compiling Numba LLVM code for the first time...")
	print("This can be very slow! Please be patient.")
	print("You never have to compile code after this one time.")

from . import base
from . import cfuncs
from . import linalg
from . import random
from . import stats
from . import utils

from . import randomized
from . import numba
