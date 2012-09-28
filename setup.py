import os
import sys
import platform
from os.path import join
from distutils.core import setup, Extension

from numba import minivect

import numpy
from Cython.Distutils import build_ext
from Cython.Distutils.extension import Extension as CythonExtension


OMP_ARGS = ['-fopenmp']
OMP_LINK = OMP_ARGS

miniutils = minivect.get_include() + "/miniutils"
miniutils_dep = miniutils + ".pyx"
miniutils_header_dep = miniutils + ".h"

ext_modules = [
    Extension(
        name = "numbapro._internal",
        sources = ["numbapro/_internal.c", "numbapro/_ufunc.c", "numbapro/_gufunc.c"],
        include_dirs = [numpy.get_include(), minivect.get_include()],
        depends = ["numbapro/_internal.h", miniutils_header_dep],
    ),

    CythonExtension(
        name = "numbapro.dispatch",
        sources = ["numbapro/dispatch.pyx"],
        include_dirs = [numpy.get_include()],
        depends = [miniutils_dep, "numbapro/dispatch.pxd"],
        extra_compile_args = OMP_ARGS,
        extra_link_args = OMP_LINK,
        cython_gdb=True,
    ),

    CythonExtension(
        name = "numbapro._minidispatch",
        sources = ["numbapro/_minidispatch.pyx"],
        depends = [miniutils_dep, "numbapro/dispatch.pxd"],
        include_dirs = [numpy.get_include(), minivect.get_include()],
        cython_include_dirs = [minivect.get_include()],
        cython_gdb=True,
    ),
]

CUDA_DIR = os.environ.get('CUDA_DIR', '/usr/local/cuda')
if os.path.exists(CUDA_DIR):
    CUDA_INCLUDE = join(CUDA_DIR, 'include')
    if sys.platform == 'linux2' and platform.architecture()[0] == '64bit':
        CUDA_LIB_DIR = join(CUDA_DIR, 'lib64')
    else:
        CUDA_LIB_DIR = join(CUDA_DIR, 'lib')

    ext = CythonExtension(
        name = "numbapro._cudadispatch",
        sources = ["numbapro/_cudadispatch.pyx", "numbapro/_cuda.c"],
        include_dirs = [numpy.get_include(), CUDA_INCLUDE],
        # extra_objects = ["numbapro/_cuda.o"],
        library_dirs = [CUDA_LIB_DIR],
        libraries = ["cuda", "cudart"],
        depends = ["numbapro/_cuda.h", "numbapro/cuda.pxd",
                   "numbapro/dispatch.pxd"],
        cython_gdb=True,
    )
    ext_modules.append(ext)

setup(
    name = "numbapro",
    author = "Continuum Analytics, Inc.",
    author_email = "support@continuum.io",
    url = "http://www.continuum.io",
    license = "Proprietary",
    description = "compile Python code",
    ext_modules = ext_modules,
    packages = ['numbapro', 'llvm_cbuilder', 'numbapro.vectorize',
                'numbapro.tests',
                'numbapro.tests.basic_vectorize',
                'numbapro.tests.llvm_cbuilder_tests',
                'numbapro.tests.parallel_vectorize',
                'numbapro.tests.stream_vectorize'],
    version = "0.5",
    cmdclass={'build_ext': build_ext},
)
