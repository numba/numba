import os
import sys
from fnmatch import fnmatchcase
from distutils.util import convert_path
from distutils.core import setup, Extension

import numpy

# import numba
import gen_type_conversion

from Cython.Distutils import build_ext
from Cython.Distutils.extension import Extension as CythonExtension

if sys.version_info[:2] < (2, 6):
    raise Exception('numba requires Python 2.6 or greater.')

import versioneer


versioneer.versionfile_source = 'numba/_version.py'
versioneer.versionfile_build = 'numba/_version.py'
versioneer.tag_prefix = ''
versioneer.parentdir_prefix = 'numba-'


cmdclass = versioneer.get_cmdclass()
cmdclass['build_ext'] = build_ext

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
    cmdclass["build_py"] = build_py
    # cmdclass["build"] = build_py

    # Distribute options
    # setup_args["use_2to3"] = True

if sys.version_info[0] >= 3:
    run_2to3()

def get_include():
    """Use numba.get_include() instead (make numba importable without
    building it first)
    """
    numba_root = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(numba_root, "numba", "include")

numba_include_dir = get_include()

gen_type_conversion.run()

setup(
    name = "numba",
    version = versioneer.get_version(),
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
        'numba' : ['include/*'],
    },
    ext_modules = [
        Extension(
            name = "numba.external.utilities.utilities",
            sources = ["numba/external/utilities/utilities.c"],
            include_dirs=[numba_include_dir],
            depends=["numba/external/utilities/type_conversion.c",
                     "numba/external/utilities/generated_conversions.c",
                     "numba/external/utilities/generated_conversions.h"]),
        CythonExtension(
            name = "numba.extension_types",
            sources = ["numba/extension_types.pyx", "numba/numbafunction.c"],
            depends = ["numba/numbafunction.h"],
            include_dirs=[numba_include_dir],
            cython_gdb=True),
        CythonExtension(
            name = "numba.numbawrapper",
            sources = ["numba/numbawrapper.pyx"],
            include_dirs=[numpy.get_include()],
            cython_gdb=True),
    ],
    cmdclass = cmdclass,
    **setup_args
)
