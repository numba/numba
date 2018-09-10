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
from distutils import sysconfig
import sys
import os
import platform

import versioneer

if sys.platform.startswith('linux'):
    # Patch for #2555 to make wheels without libpython
    sysconfig.get_config_vars()['Py_ENABLE_SHARED'] = 0


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
    CFLAGS = ['-g']

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


    ext_npyufunc_workqueue_impls = []

    def check_file_at_path(path2file):
        """
        Takes a list as a path, a single glob (*) is permitted as an entry which
        indicates that expansion at this location is required (i.e. version
        might not be known).
        """
        found = None
        path2check = [os.path.split(os.path.split(sys.executable)[0])[0]]
        path2check += [os.getenv(n, '') for n in ['CONDA_PREFIX', 'PREFIX']]
        if sys.platform.startswith('win'):
            path2check += [os.path.join(p, 'Library') for p in path2check]
        for p in path2check:
            if p:
                if '*' in path2file:
                    globloc = path2file.index('*')
                    searchroot = os.path.join(*path2file[:globloc])
                    try:
                        potential_locs = os.listdir(os.path.join(p, searchroot))
                    except BaseException:
                        continue
                    searchfor = path2file[globloc + 1:]
                    for x in potential_locs:
                        potpath = os.path.join(p, searchroot, x, *searchfor)
                        if os.path.isfile(potpath):
                            found = p  # the latest is used
                elif os.path.isfile(os.path.join(p, *path2file)):
                    found = p  # the latest is used
        return found

    # Search for Intel TBB, first check env var TBBROOT then conda locations
    tbb_root = os.getenv('TBBROOT')
    if not tbb_root:
        tbb_root = check_file_at_path(['include', 'tbb', 'tbb.h'])

    # Set various flags for use in TBB and openmp. On OSX, also find OpenMP!
    have_openmp = True
    if sys.platform.startswith('win'):
        cpp11flags = []
        ompcompileflags = ['-openmp']
        omplinkflags = []
    elif sys.platform.startswith('darwin'):
        cpp11flags = ['-std=c++11']
        # This is a bit unusual but necessary...
        # llvm (clang) OpenMP is used for headers etc at compile time
        # Intel OpenMP (libiomp5) provides the link library.
        # They are binary compatible and may not safely coexist in a process, as
        # libiomp5 is more prevalent and often linked in for NumPy it is used
        # here!
        ompcompileflags = ['-fopenmp']
        omplinkflags = ['-fopenmp=libiomp5']
        omppath = ['lib', 'clang', '*', 'include', 'omp.h']
        have_openmp = check_file_at_path(omppath)
    else:
        cpp11flags = ['-std=c++11']
        ompcompileflags = ['-fopenmp']
        if platform.machine() == 'ppc64le':
            omplinkflags = ['-fopenmp']
        else:
            omplinkflags = ['-fopenmp']

    if tbb_root:
        print("Using Intel TBB from:", tbb_root)
        ext_npyufunc_tbb_workqueue = Extension(
            name='numba.npyufunc.tbbpool',
            sources=['numba/npyufunc/tbbpool.cpp', 'numba/npyufunc/gufunc_scheduler.cpp'],
            depends=['numba/npyufunc/workqueue.h'],
            include_dirs=[os.path.join(tbb_root, 'include')],
            extra_compile_args=cpp11flags,
            libraries   =['tbb'],  # TODO: if --debug or -g, use 'tbb_debug'
            library_dirs=[os.path.join(tbb_root, 'lib', 'intel64', 'gcc4.4'),  # for Linux
                        os.path.join(tbb_root, 'lib'),                       # for MacOS
                        os.path.join(tbb_root, 'lib', 'intel64', 'vc_mt'),   # for Windows
                        ],
            )
        ext_npyufunc_workqueue_impls.append(ext_npyufunc_tbb_workqueue)
    else:
        print("TBB not found")

    if have_openmp:
        print("Using OpenMP from:", have_openmp)
        # OpenMP backed work queue
        ext_npyufunc_omppool = Extension( name='numba.npyufunc.omppool',
                                    sources=['numba/npyufunc/omppool.cpp',
                                            'numba/npyufunc/gufunc_scheduler.cpp'],
                                    depends=['numba/npyufunc/workqueue.h'],
                                    extra_compile_args=ompcompileflags + cpp11flags,
                                    extra_link_args = omplinkflags)

        ext_npyufunc_workqueue_impls.append(ext_npyufunc_omppool)
    else:
        print("OpenMP not found")

    # Build the Numba workqueue implementation irrespective of whether the TBB
    # version is built. Users can select a backend via env vars.
    ext_npyufunc_workqueue = Extension(
        name='numba.npyufunc.workqueue',
        sources=['numba/npyufunc/workqueue.c', 'numba/npyufunc/gufunc_scheduler.cpp'],
        depends=['numba/npyufunc/workqueue.h'])
    ext_npyufunc_workqueue_impls.append(ext_npyufunc_workqueue)


    ext_mviewbuf = Extension(name='numba.mviewbuf',
                             extra_link_args=install_name_tool_fixer,
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

    ext_cuda_extras = Extension(name='numba.cuda.cudadrv._extras',
                                sources=['numba/cuda/cudadrv/_extras.c'],
                                depends=['numba/_pymodule.h'],
                                include_dirs=["numba"])

    ext_modules = [ext_dynfunc, ext_dispatcher, ext_helperlib, ext_typeconv,
                   ext_npyufunc_ufunc, ext_mviewbuf, ext_nrt_python,
                   ext_jitclass_box, ext_cuda_extras]

    ext_modules += ext_npyufunc_workqueue_impls

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

install_requires = ['llvmlite>=0.25.0dev0', 'numpy']
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
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Topic :: Software Development :: Compilers",
    ],
    package_data={
        # HTML templates for type annotations
        "numba.annotations": ["*.html"],
        # Various test data
        "numba.cuda.tests.cudadrv.data": ["*.ptx"],
        "numba.tests": ["pycc_distutils_usecase/*.py"],
        # Some C files are needed by pycc
        "numba": ["*.c", "*.h"],
        "numba.pycc": ["*.c", "*.h"],
        "numba.runtime": ["*.c", "*.h"],
    },
    scripts=["numba/pycc/pycc", "bin/numba"],
    author="Anaconda, Inc.",
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
