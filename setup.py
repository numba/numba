try:
    # Try to use setuptools so as to enable support of the special
    # "Microsoft Visual C++ Compiler for Python 2.7" (http://aka.ms/vcpython27)
    # for building under Windows.
    # Note setuptools >= 6.0 is required for this.
    from setuptools import setup, Extension
except ImportError:
    from distutils.core import setup, Extension

from distutils.command import build
from distutils.spawn import spawn
import sys
import os

import versioneer


class build_doc(build.build):
    description = "build documentation"

    def run(self):
        spawn(['make', '-C', 'docs', 'html'])


versioneer.VCS = 'git'
versioneer.versionfile_source = 'numba/_version.py'
versioneer.versionfile_build = 'numba/_version.py'
versioneer.tag_prefix = ''
versioneer.parentdir_prefix = 'numba-'

cmdclass = versioneer.get_cmdclass()
cmdclass['build_doc'] = build_doc


GCCFLAGS = ["-std=c89", "-Wdeclaration-after-statement", "-Werror"]

if os.environ.get("NUMBA_GCC_FLAGS"):
    CFLAGS = GCCFLAGS
else:
    CFLAGS = []

install_name_tool_fixer = []
if sys.platform == 'darwin':
    install_name_tool_fixer += ['-headerpad_max_install_names']


def is_building():
    """
    Parse the setup.py command and return whether a build is requested.
    If False is returned, only an informational command is run.
    If True is returned, information about C extensions will have to
    be passed to the setup() function.
    """
    if len(sys.argv) < 2:
        # User forgot to give an argument probably, let setuptools handle that.
        return True

    info_commands = ['--help-commands', '--name', '--version', '-V',
                     '--fullname', '--author', '--author-email',
                     '--maintainer', '--maintainer-email', '--contact',
                     '--contact-email', '--url', '--license', '--description',
                     '--long-description', '--platforms', '--classifiers',
                     '--keywords', '--provides', '--requires', '--obsoletes']
    # Add commands that do more than print info, but also don't need
    # any build step.
    info_commands.extend(['egg_info', 'install_egg_info', 'rotate'])

    for command in info_commands:
        if command in sys.argv[1:]:
            return False

    return True


def get_ext_modules():
    """
    Return a list of Extension instances for the setup() call.
    """
    # Note we don't import Numpy at the toplevel, since setup.py
    # should be able to run without Numpy for pip to discover the
    # build dependencies
    import numpy.distutils.misc_util as np_misc

    # Inject required options for extensions compiled against the Numpy
    # C API (include dirs, library dirs etc.)
    np_compile_args = np_misc.get_info('npymath')

    ext_dynfunc = Extension(name='numba._dynfunc',
                            sources=['numba/_dynfuncmod.c'],
                            extra_compile_args=CFLAGS,
                            depends=['numba/_pymodule.h',
                                     'numba/_dynfunc.c'])

    ext_dispatcher = Extension(name="numba._dispatcher",
                               sources=['numba/_dispatcher.c',
                                        'numba/_typeof.c',
                                        'numba/_hashtable.c',
                                        'numba/_dispatcherimpl.cpp',
                                        'numba/typeconv/typeconv.cpp'],
                               depends=["numba/_pymodule.h",
                                        "numba/_dispatcher.h",
                                        "numba/_typeof.h",
                                        "numba/_hashtable.h"],
                               **np_compile_args)

    ext_helperlib = Extension(name="numba._helperlib",
                              sources=["numba/_helpermod.c",
                                       "numba/_math_c99.c"],
                              extra_compile_args=CFLAGS,
                              extra_link_args=install_name_tool_fixer,
                              depends=["numba/_pymodule.h",
                                       "numba/_math_c99.h",
                                       "numba/_helperlib.c",
                                       "numba/_lapack.c",
                                       "numba/_npymath_exports.c",
                                       "numba/_random.c",
                                       "numba/mathnames.inc"],
                              **np_compile_args)

    ext_typeconv = Extension(name="numba.typeconv._typeconv",
                             sources=["numba/typeconv/typeconv.cpp",
                                      "numba/typeconv/_typeconv.cpp"],
                             depends=["numba/_pymodule.h"],
                             )

    ext_npyufunc_ufunc = Extension(name="numba.npyufunc._internal",
                                   sources=["numba/npyufunc/_internal.c"],
                                   depends=["numba/npyufunc/_ufunc.c",
                                            "numba/npyufunc/_internal.h",
                                            "numba/_pymodule.h"],
                                   **np_compile_args)

    ext_npyufunc_workqueue = Extension(
        name='numba.npyufunc.workqueue',
        sources=['numba/npyufunc/workqueue.c'],
        depends=['numba/npyufunc/workqueue.h'])


    ext_mviewbuf = Extension(name='numba.mviewbuf',
                             sources=['numba/mviewbuf.c'])

    ext_nrt_python = Extension(name='numba.runtime._nrt_python',
                               sources=['numba/runtime/_nrt_pythonmod.c',
                                        'numba/runtime/nrt.c'],
                               depends=['numba/runtime/nrt.h',
                                        'numba/_pymodule.h',
                                        'numba/runtime/_nrt_python.c'],
                               **np_compile_args)

    ext_jitclass_box = Extension(name='numba.jitclass._box',
                                 sources=['numba/jitclass/_box.c'],
                                 depends=['numba/_pymodule.h'],
                                 )

    ext_modules = [ext_dynfunc, ext_dispatcher,
                   ext_helperlib, ext_typeconv,
                   ext_npyufunc_ufunc, ext_npyufunc_workqueue, ext_mviewbuf,
                   ext_nrt_python, ext_jitclass_box]

    return ext_modules


def find_packages(root_dir, root_name):
    """
    Recursively find packages in *root_dir*.
    """
    packages = []
    def rec(path, pkg_name):
        packages.append(pkg_name)
        for fn in sorted(os.listdir(path)):
            subpath = os.path.join(path, fn)
            if os.path.exists(os.path.join(subpath, "__init__.py")):
                subname = "%s.%s" % (pkg_name, fn)
                rec(subpath, subname)
    rec(root_dir, root_name)
    return packages

packages = find_packages("numba", "numba")


build_requires = ['numpy']

install_requires = ['llvmlite', 'numpy']
if sys.version_info < (3, 4):
    install_requires.extend(['enum34', 'singledispatch'])
if sys.version_info < (3, 3):
    install_requires.append('funcsigs')

metadata = dict(
    name='numba',
    description="compiling Python code using LLVM",
    version=versioneer.get_version(),

    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3.4",
        "Programming Language :: Python :: 3.5",
        "Topic :: Software Development :: Compilers",
    ],
    package_data={
        # HTML templates for type annotations
        "numba.annotations": ["*.html"],
        # Various test data
        "numba.cuda.tests.cudadrv.data": ["*.ptx"],
        "numba.hsa.tests.hsadrv": ["*.brig"],
        "numba.tests": ["pycc_distutils_usecase/*.py"],
        # Some C files are needed by pycc
        "numba": ["*.c", "*.h"],
        "numba.pycc": ["*.c", "*.h"],
        "numba.runtime": ["*.c", "*.h"],
    },
    scripts=["numba/pycc/pycc", "bin/numba"],
    author="Continuum Analytics, Inc.",
    author_email="numba-users@continuum.io",
    url="http://numba.github.com",
    packages=packages,
    setup_requires=build_requires,
    install_requires=install_requires,
    license="BSD",
    cmdclass=cmdclass,
    )

with open('README.rst') as f:
    metadata['long_description'] = f.read()

if is_building():
    metadata['ext_modules'] = get_ext_modules()

setup(**metadata)
