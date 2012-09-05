from distutils.core import setup, Extension
import numpy
from numba import minivect

from Cython.Distutils import build_ext
from Cython.Distutils.extension import Extension as CythonExtension

setup(
    name = "numbapro",
    author = "Continuum Analytics, Inc.",
    author_email = "support@continuum.io",
    url = "http://www.continuum.io",
    license = "Proprietary",
    description = "compile Python code",
    ext_modules = [
        Extension(
            name = "numbapro._internal",
            sources = ["numbapro/_internal.c"],
            include_dirs = [numpy.get_include(), minivect.get_include()]),
        CythonExtension(
            name = "numbapro._minidispatch",
            sources = ["numbapro/_minidispatch.pyx"],
            include_dirs = [numpy.get_include()]),
    ],
    packages = ['numbapro', 'llvm_cbuilder', 'numbapro.vectorize',
                'numbapro.tests',
                'numbapro.tests.basic_vectorize',
                'numbapro.tests.llvm_cbuilder_tests',
                'numbapro.tests.parallel_vectorize',
                'numbapro.tests.stream_vectorize'],
    version = "0.5",
    cmdclass={'build_ext': build_ext},
)
