
print("******* Now compiling Numba and LLVM code..... *******")
print("******* This can be VERY SLOW. Please wait.... *******\n"
	"Progress: |||||||||||||||", end = "")

from hyperlearn.numba.funcs import *

print("|||||||||||||||", end = "")

from hyperlearn.utils import *

print("|||||||||||||||")

from hyperlearn.stats import *

print("******* Code has been successfully compiled!:) *******")
