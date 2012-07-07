import re
import sys
from os.path import join
from distutils.core import setup, Extension
import numpy

if sys.version_info[:2] < (2, 5):
    raise Exception('numba requires Python 2.5 or greater.')

kwds = {}

kwds['long_description'] = open('README').read()


setup(
    name = "numba",
    author = "Travis Oliphant",
    author_email = "travis@continuum.io",
    url = "https://github.com/ContinuumIO/numba",
    license = "BSD",
    classifiers = [
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Programming Language :: Python :: 2.6",
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3.2",
        "Topic :: Utilities",
    ],
    description = "compiling Python code for NumPy",
    packages = ["numba",
                "numba.pymothoa",
                "numba.pymothoa.util",
                "numba.minivect"],
    ext_modules = [Extension(name = "numba._ext",
                             sources = ["numba/_ext.c"],
                             include_dirs=[numpy.get_include()])],
    version = '0.1'
)
