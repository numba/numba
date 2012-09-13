import os
import sys
import subprocess
from distutils import sysconfig
from distutils.core import setup, Extension

from numba import minivect

import numpy
from Cython.Distutils import build_ext
from Cython.Distutils.extension import Extension as CythonExtension


def search_on_path(filename):
    """Find file on system path."""
    # http://aspn.activestate.com/ASPN/Cookbook/Python/Recipe/52224
    search_path = os.environ["PATH"]

    paths = search_path.split(os.pathsep)
    for path in paths:
        if os.path.exists(os.path.join(path, filename)):
            return os.path.abspath(os.path.join(path, filename))


OMP_ARGS = ['-fopenmp']
OMP_LINK = OMP_ARGS

ext_modules = [
    Extension(
        name = "numbapro._internal",
        sources = ["numbapro/_internal.c"],
        include_dirs = [numpy.get_include(), minivect.get_include()]
    ),

    CythonExtension(
        name = "numbapro.utils",
        sources = ["numbapro/utils.pyx"],
        include_dirs = [numpy.get_include()],
        # extra_objects = ["numbapro/_cuda.o"],
    ),

    CythonExtension(
        name = "numbapro._minidispatch",
        sources = ["numbapro/_minidispatch.pyx"],
        depends = [minivect.get_include() + "/miniutils.pyx"],
        include_dirs = [numpy.get_include(), minivect.get_include()],
        cython_include_dirs = [minivect.get_include()],
        extra_compile_args = OMP_ARGS,
        extra_link_args = OMP_LINK,
    ),
]

nvcc_path = search_on_path("nvcc")
if nvcc_path is not None:
    CUDA_ROOT = os.path.dirname(os.path.dirname(nvcc_path))
    CUDA_LIB_DIR = os.path.join(CUDA_ROOT, 'lib')
    CUDA_INCLUDE = os.path.join(CUDA_ROOT, 'include')

    if sys.maxint > 2 ** 31 and os.path.exists(CUDA_LIB_DIR + '64'):
        CUDA_LIB_DIR += '64'

    ext = CythonExtension(
        name = "numbapro._cudadispatch",
        sources = ["numbapro/_cudadispatch.pyx", "numbapro/_cuda.c"],
        include_dirs = [numpy.get_include(), CUDA_INCLUDE],
        # extra_objects = ["numbapro/_cuda.o"],
        library_dirs = [CUDA_LIB_DIR],
        libraries = ["cuda", "cudart"],
        depends = ["numbapro/_cuda.h", "numbapro/cuda.pxd", "numbapro/utils.pxd"],
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
