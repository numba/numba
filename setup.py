import os
import sys
from fnmatch import fnmatchcase
from distutils.util import convert_path
from distutils.core import setup, Extension

import numpy


if sys.version_info[:2] < (2, 5):
    raise Exception('numba requires Python 2.5 or greater.')

kwds = {}

kwds['long_description'] = open('README').read()


def find_packages(where='.', exclude=()):
    out = []
    stack=[(convert_path(where), '')]
    while stack:
        where, prefix = stack.pop(0)
        for name in os.listdir(where):
            fn = os.path.join(where,name)
            if ('.' not in name and os.path.isdir(fn) and
                os.path.isfile(os.path.join(fn, '__init__.py'))
            ):
                out.append(prefix+name)
                stack.append((fn, prefix+name+'.'))
    for pat in list(exclude) + ['ez_setup', 'distribute_setup']:
        out = [item for item in out if not fnmatchcase(item, pat)]
    return out

setup(
    name = "numba",
    author = "Continuum Analytics, Inc.",
    author_email = "numba-users@continuum.io",
    url = "http://numba.github.com",
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
    description = "compiling Python code using LLVM",
    packages = find_packages(),
    scripts = ['numba/pycc/pycc'],
    package_data = {
        'numba.minivect' : ['include/*'],
    },
    # ext_modules = [Extension(name = "numba._ext",
    #                          sources = ["numba/_ext.c"],
    #                          include_dirs=[numpy.get_include()])],
    version = '0.2'
)
