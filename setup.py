# NOTE: for building under Windows.
# Use setuptools so as to enable support of the special
# "Microsoft Visual C++ Compiler for Python 2.7" (http://aka.ms/vcpython27)
# Note setuptools >= 6.0 is required for this.
from setuptools import setup, Extension, find_packages
from distutils.command import build
from distutils.spawn import spawn
from distutils import sysconfig
import sys
import os
import platform

import versioneer

min_python_version = "3.6"
min_numpy_build_version = "1.11"
min_numpy_run_version = "1.15"
min_llvmlite_version = "0.31.0dev0"
max_llvmlite_version = "0.32.0"

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


def is_building_wheel():
    if len(sys.argv) < 2:
        # No command is given.
        return False

    return 'bdist_wheel' in sys.argv[1:]


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
                                        'numba/core/typeconv/typeconv.cpp'],
                               depends=["numba/_pymodule.h",
                                        "numba/_dispatcher.h",
                                        "numba/_typeof.h",
                                        "numba/_hashtable.h"],
                               **np_compile_args)

    ext_helperlib = Extension(name="numba._helperlib",
                              sources=["numba/_helpermod.c",
                                       "numba/cext/utils.c",
                                       "numba/cext/dictobject.c",
                                       "numba/cext/listobject.c",
                                       ],
                              extra_compile_args=CFLAGS,
                              extra_link_args=install_name_tool_fixer,
                              depends=["numba/_pymodule.h",
                                       "numba/_helperlib.c",
                                       "numba/_lapack.c",
                                       "numba/_npymath_exports.c",
                                       "numba/_random.c",
                                       "numba/mathnames.inc",
                                       ],
                              **np_compile_args)

    ext_typeconv = Extension(name="numba.core.typeconv._typeconv",
                             sources=["numba/core/typeconv/typeconv.cpp",
                                      "numba/core/typeconv/_typeconv.cpp"],
                             depends=["numba/_pymodule.h"],
                             )

    ext_np_ufunc = Extension(name="numba.np.ufunc._internal",
                             sources=["numba/np/ufunc/_internal.c"],
                             depends=["numba/np/ufunc/_ufunc.c",
                                     "numba/np/ufunc/_internal.h",
                                     "numba/_pymodule.h"],
                             **np_compile_args)

    ext_np_ufunc_backends = []

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
        ext_np_ufunc_tbb_backend = Extension(
            name='numba.np.ufunc.tbbpool',
            sources=[
                'numba/np/ufunc/tbbpool.cpp',
                'numba/np/ufunc/gufunc_scheduler.cpp',
            ],
            depends=['numba/np/ufunc/workqueue.h'],
            include_dirs=[os.path.join(tbb_root, 'include')],
            extra_compile_args=cpp11flags,
            libraries=['tbb'],  # TODO: if --debug or -g, use 'tbb_debug'
            library_dirs=[
                # for Linux
                os.path.join(tbb_root, 'lib', 'intel64', 'gcc4.4'),
                # for MacOS
                os.path.join(tbb_root, 'lib'),
                # for Windows
                os.path.join(tbb_root, 'lib', 'intel64', 'vc_mt'),
            ],
        )
        ext_np_ufunc_backends.append(ext_np_ufunc_tbb_backend)
    else:
        print("TBB not found")

    # Disable OpenMP if we are building a wheel or
    # forced by user with NUMBA_NO_OPENMP=1
    if is_building_wheel() or os.getenv('NUMBA_NO_OPENMP'):
        print("OpenMP disabled")
    elif have_openmp:
        print("Using OpenMP from:", have_openmp)
        # OpenMP backed work queue
        ext_np_ufunc_omppool_backend = Extension(
            name='numba.np.ufunc.omppool',
            sources=[
                'numba/np/ufunc/omppool.cpp',
                'numba/np/ufunc/gufunc_scheduler.cpp',
            ],
            depends=['numba/np/ufunc/workqueue.h'],
            extra_compile_args=ompcompileflags + cpp11flags,
            extra_link_args=omplinkflags,
        )

        ext_np_ufunc_backends.append(ext_np_ufunc_omppool_backend)
    else:
        print("OpenMP not found")

    # Build the Numba workqueue implementation irrespective of whether the TBB
    # version is built. Users can select a backend via env vars.
    ext_np_ufunc_workqueue_backend = Extension(
        name='numba.np.ufunc.workqueue',
        sources=['numba/np/ufunc/workqueue.c',
                 'numba/np/ufunc/gufunc_scheduler.cpp'],
        depends=['numba/np/ufunc/workqueue.h'])
    ext_np_ufunc_backends.append(ext_np_ufunc_workqueue_backend)

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

    ext_jitclass_box = Extension(name='numba.experimental.jitclass._box',
                                 sources=['numba/experimental/jitclass/_box.c'],
                                 depends=['numba/experimental/_pymodule.h'],
                                 )

    ext_cuda_extras = Extension(name='numba.cuda.cudadrv._extras',
                                sources=['numba/cuda/cudadrv/_extras.c'],
                                depends=['numba/_pymodule.h'],
                                include_dirs=["numba"])

    ext_modules = [ext_dynfunc, ext_dispatcher, ext_helperlib, ext_typeconv,
                   ext_np_ufunc, ext_mviewbuf, ext_nrt_python,
                   ext_jitclass_box, ext_cuda_extras]

    ext_modules += ext_np_ufunc_backends

    return ext_modules


packages = find_packages(include=["numba", "numba.*"])

build_requires = [f'numpy >={min_numpy_build_version}']
install_requires = [
    f'llvmlite >={min_llvmlite_version},<{max_llvmlite_version}',
    f'numpy >={min_numpy_run_version}',
    'setuptools',
]

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
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
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
        "numba.cext": ["*.c", "*.h"],
        # numba gdb hook init command language file
        "numba.targets": ["cmdlang.gdb"],
    },
    scripts=["numba/pycc/pycc", "bin/numba"],
    author="Anaconda, Inc.",
    author_email="numba-users@continuum.io",
    url="http://numba.github.com",
    packages=packages,
    setup_requires=build_requires,
    install_requires=install_requires,
    python_requires=f">={min_python_version}",
    license="BSD",
    cmdclass=cmdclass,
)

with open('README.rst') as f:
    metadata['long_description'] = f.read()

if is_building():
    metadata['ext_modules'] = get_ext_modules()

setup(**metadata)
