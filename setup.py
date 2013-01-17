import os
import sys
from fnmatch import fnmatchcase
from distutils.util import convert_path
from distutils.core import setup, Extension

import numpy

from Cython.Distutils import build_ext
from Cython.Distutils.extension import Extension as CythonExtension

if sys.version_info[:2] < (2, 7):
    raise Exception('numba requires Python 2.7 or greater.')

cmdclasses = {
    'build_ext': build_ext,
}

setup_args = {
    'long_description': open('README').read(),
}

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

def run_2to3():
    import lib2to3.refactor
    from distutils.command.build_py import build_py_2to3 as build_py
    print("Installing 2to3 fixers")
    # need to convert sources to Py3 on installation
    fixes = lib2to3.refactor.get_fixers_from_package("lib2to3.fixes")
    bad_fixers = ('next', 'funcattrs')
    fixes = [fix for fix in fixes
              if fix.split('fix_')[-1] not in bad_fixers]

    build_py.fixer_names = fixes
    cmdclasses["build_py"] = build_py
    # cmdclasses["build"] = build_py

    # Distribute options
    # setup_args["use_2to3"] = True

if sys.version_info[0] >= 3:
    run_2to3()

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
        # "Programming Language :: Python :: 2.6",
        "Programming Language :: Python :: 2.7",
        # "Programming Language :: Python :: 3.2",
        "Topic :: Utilities",
    ],
    description = "compiling Python code using LLVM",
    packages = find_packages(),
    scripts = ['numba/pycc/pycc'],
    package_data = {
        'numba.minivect' : ['include/*'],
        'numba.asdl.common': ['*.asdl'],
        'numba.asdl.py2_7': ['*.asdl'],
    },
    ext_modules = [
#        Extension(name = "numba._ext",
#                  sources = ["numba/_ext.c"],
#                  include_dirs=[numpy.get_include()]),
        CythonExtension(
            name = "numba.extension_types",
            sources = ["numba/extension_types.pyx", "numba/numbafunction.c"],
            cython_gdb=True),
    ],
    version = '0.5.0',
    cmdclass=cmdclasses,
    **setup_args
)
