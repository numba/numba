import os
import sys
import platform
from os.path import join
from distutils.core import setup, Extension

from numba import minivect

import numpy
from Cython.Distutils import build_ext
from Cython.Distutils.extension import Extension as CythonExtension

# NOTE: On OSX 10.8.2, XCode 4.2 you need -D_FORTIFY_SOURCE=0
#       http://comments.gmane.org/gmane.comp.compilers.llvm.devel/37989
#       This is a problem with GCC builtin.
#       And you need to export CC=gcc, but there is still problem at linkage,
#       where it uses clang.  I am solving that manually by copying the
#       the commands and swap out clang with gcc.
OMP_ARGS = ['-fopenmp']
OMP_LINK = OMP_ARGS

miniutils = minivect.get_include() + "/miniutils"
miniutils_dep = miniutils + ".pyx"
miniutils_header_dep = miniutils + ".h"

ext_modules = [
    Extension(
        name = "numba.vectorize._internal",
        sources = ["numba_vectorize/_internal.c",
                   "numba_vectorize/_ufunc.c",
                   "numba_vectorize/_gufunc.c"],
        include_dirs = [numpy.get_include(), minivect.get_include()],
        depends = ["numba_vectorize/_internal.h", miniutils_header_dep],
    ),

    CythonExtension(
        name = "numbapro._minidispatch",
        sources = ["numbapro/_minidispatch.pyx"],
        depends = [miniutils_dep, "numbapro/dispatch.pxd"],
        include_dirs = [numpy.get_include(), minivect.get_include()],
        cython_include_dirs = [minivect.get_include()],
        cython_gdb=True,
    ),
               
    CythonExtension(
        name = "numbapro.dispatch",
        sources = ["numbapro/dispatch.pyx"],
        include_dirs = [numpy.get_include(), "numba_vectorize"],
        depends = [miniutils_dep, "numbapro/dispatch.pxd"],
        extra_compile_args = OMP_ARGS + ['-D_FORTIFY_SOURCE=0'],
        extra_link_args = OMP_LINK,
        cython_gdb=True,
    ),

#    CythonExtension(
#        name = "numbapro._cudadispatchlib",
#        sources = ["numbapro/_cudadispatchlib.pyx", "numbapro/_cuda.c"],
#        include_dirs = [numpy.get_include(), "cuda_toolkit"],
#        depends = ["numbapro/_cuda.h", "numbapro/cuda.pxd",
#                   "numbapro/dispatch.pxd",],
#        cython_gdb=True,
#    )
]

setup(
    name = "numbapro",
    author = "Continuum Analytics, Inc.",
    author_email = "support@continuum.io",
    url = "http://www.continuum.io",
    license = "Proprietary",
    description = "compile Python code",
    ext_modules = ext_modules,
      package_dir = {'numba.vectorize': 'numba_vectorize'},
    packages = ['numbapro', 'numbapro.vectorize',
                'numbapro._cuda',
                'numbapro._utils',
                'numbapro.tests',
                'numbapro.tests.basic_vectorize',
                'numbapro.tests.parallel_vectorize',
                'numbapro.tests.stream_vectorize',
                'numbapro.tests.cuda', 'numbapro.tests.cuda.fail'] + [
                'numba.vectorize'
                ],
    version = "0.7.3",
    cmdclass={'build_ext': build_ext},
)
