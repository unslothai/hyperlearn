

from distutils.core import setup
from Cython.Build import cythonize
from numpy import get_include
from multiprocessing import cpu_count

# kwargs = {
#     "ext_modules" : [
#         Extension(
#             "base", ["./base.c"]
#             ),
#         Extension(
#             "utils", ["./utils.c"]
#             )
#         ],
#     "include_dirs" : [get_include()],
# }

# check = os.listdir("./hyperlearn.")
# print(check)
# USE_CYTHON = True

# for x in check:
#     if ".c" in x:
#         USE_CYTHON = False
#         break


# if USE_CYTHON:
#     kwargs["ext_modules"] = \
#         cythonize("../*.pyx",
#             compiler_directives = {
#                 'language_level':3, 
#                 'boundscheck':False, 
#                 'wraparound':False,
#                 'initializedcheck':False, 
#                 'cdivision':True,
#                 'nonecheck':False,
#             },
#         )

#setup(**kwargs)

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
        nthreads = cpu_count()- 1,
        quiet = True,
    ),
    include_dirs = [get_include()]
)
