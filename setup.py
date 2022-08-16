import os
import platform
import sys
from distutils import sysconfig
from distutils.command import build
from distutils.command.build_ext import build_ext
from distutils.spawn import spawn

from setuptools import Extension, find_packages, setup
import versioneer

_version_module = None
try:
    from packaging import version as _version_module
except ImportError:
    try:
        from setuptools._vendor.packaging import version as _version_module
    except ImportError:
        pass


min_python_version = "3.7"
max_python_version = "3.11"  # exclusive
min_numpy_build_version = "1.11"
min_numpy_run_version = "1.18"
max_numpy_run_version = "1.24"
min_llvmlite_version = "0.39.0dev0"
max_llvmlite_version = "0.40"

if sys.platform.startswith('linux'):
    # Patch for #2555 to make wheels without libpython
    sysconfig.get_config_vars()['Py_ENABLE_SHARED'] = 0


def _guard_py_ver():
    if _version_module is None:
        return

    parse = _version_module.parse

    min_py = parse(min_python_version)
    max_py = parse(max_python_version)
    cur_py = parse('.'.join(map(str, sys.version_info[:3])))

    if not min_py <= cur_py < max_py:
        msg = ('Cannot install on Python version {}; only versions >={},<{} '
               'are supported.')
        raise RuntimeError(msg.format(cur_py, min_py, max_py))


_guard_py_ver()


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

extra_link_args = []
install_name_tool_fixer = []
if sys.platform == 'darwin':
    install_name_tool_fixer += ['-headerpad_max_install_names']
if platform.machine() == 'ppc64le':
    extra_link_args += ['-pthread']

build_ext = cmdclass.get('build_ext', build_ext)

numba_be_user_options = [
    ('werror', None, 'Build extensions with -Werror'),
    ('wall', None, 'Build extensions with -Wall'),
    ('noopt', None, 'Build extensions without optimization'),
]


class NumbaBuildExt(build_ext):

    user_options = build_ext.user_options + numba_be_user_options
    boolean_options = build_ext.boolean_options + ['werror', 'wall', 'noopt']

    def initialize_options(self):
        super().initialize_options()
        self.werror = 0
        self.wall = 0
        self.noopt = 0

    def run(self):
        extra_compile_args = []
        if self.noopt:
            if sys.platform == 'win32':
                extra_compile_args.append('/Od')
            else:
                extra_compile_args.append('-O0')
        if self.werror:
            extra_compile_args.append('-Werror')
        if self.wall:
            extra_compile_args.append('-Wall')
        for ext in self.extensions:
            ext.extra_compile_args.extend(extra_compile_args)

        super().run()


cmdclass['build_ext'] = NumbaBuildExt


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

    build_commands = ['build', 'build_py', 'build_ext', 'build_clib'
                      'build_scripts', 'install', 'install_lib',
                      'install_headers', 'install_scripts', 'install_data',
                      'sdist', 'bdist', 'bdist_dumb', 'bdist_rpm',
                      'bdist_wininst', 'check', 'build_doc', 'bdist_wheel',
                      'bdist_egg', 'develop', 'easy_install', 'test']
    return any(bc in sys.argv[1:] for bc in build_commands)


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

    ext_devicearray = Extension(name='numba._devicearray',
                                sources=['numba/_devicearray.cpp'],
                                depends=['numba/_pymodule.h',
                                         'numba/_devicearray.h'],
                                include_dirs=['numba'],
                                extra_compile_args=['-std=c++11'],
                                )

    ext_dynfunc = Extension(name='numba._dynfunc',
                            sources=['numba/_dynfuncmod.c'],
                            depends=['numba/_pymodule.h',
                                     'numba/_dynfunc.c'])

    ext_dispatcher = Extension(name="numba._dispatcher",
                               sources=['numba/_dispatcher.cpp',
                                        'numba/_typeof.c',
                                        'numba/_hashtable.c',
                                        'numba/core/typeconv/typeconv.cpp'],
                               depends=["numba/_pymodule.h",
                                        "numba/_typeof.h",
                                        "numba/_hashtable.h"],
                               **np_compile_args)

    ext_helperlib = Extension(name="numba._helperlib",
                              sources=["numba/_helpermod.c",
                                       "numba/cext/utils.c",
                                       "numba/cext/dictobject.c",
                                       "numba/cext/listobject.c",
                                       ],
                              # numba/_random.c needs pthreads
                              extra_link_args=install_name_tool_fixer +
                              extra_link_args,
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

    ext_npyufunc_num_threads = Extension(name="numba.np.ufunc._num_threads",
                                         sources=[
                                             "numba/np/ufunc/_num_threads.c"],
                                         depends=["numba/_pymodule.h"],
                                         )

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

    # Set various flags for use in TBB and openmp. On OSX, also find OpenMP!
    have_openmp = True
    if sys.platform.startswith('win'):
        if 'MSC' in sys.version:
            cpp11flags = []
            ompcompileflags = ['-openmp']
            omplinkflags = []
        else:
            # For non-MSVC toolchain e.g. gcc and clang with mingw
            cpp11flags = ['-std=c++11']
            ompcompileflags = ['-fopenmp']
            omplinkflags = ['-fopenmp']
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

    # Disable tbb if forced by user with NUMBA_DISABLE_TBB=1
    if os.getenv("NUMBA_DISABLE_TBB"):
        print("TBB disabled")
    else:
        # Search for Intel TBB, first check env var TBBROOT then conda locations
        tbb_root = os.getenv('TBBROOT')
        if not tbb_root:
            tbb_root = check_file_at_path(['include', 'tbb', 'tbb.h'])

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
                extra_link_args=extra_link_args,
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

    # Disable OpenMP if forced by user with NUMBA_DISABLE_OPENMP=1
    if os.getenv('NUMBA_DISABLE_OPENMP'):
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
        depends=['numba/np/ufunc/workqueue.h'],
        extra_link_args=extra_link_args)
    ext_np_ufunc_backends.append(ext_np_ufunc_workqueue_backend)

    ext_mviewbuf = Extension(name='numba.mviewbuf',
                             extra_link_args=install_name_tool_fixer,
                             sources=['numba/mviewbuf.c'])

    ext_nrt_python = Extension(name='numba.core.runtime._nrt_python',
                               sources=['numba/core/runtime/_nrt_pythonmod.c',
                                        'numba/core/runtime/nrt.c'],
                               depends=['numba/core/runtime/nrt.h',
                                        'numba/_pymodule.h',
                                        'numba/core/runtime/_nrt_python.c'],
                               **np_compile_args)

    ext_jitclass_box = Extension(name='numba.experimental.jitclass._box',
                                 sources=['numba/experimental/jitclass/_box.c'],
                                 depends=['numba/experimental/_pymodule.h'],
                                 )

    ext_cuda_extras = Extension(name='numba.cuda.cudadrv._extras',
                                sources=['numba/cuda/cudadrv/_extras.c'],
                                depends=['numba/_pymodule.h'],
                                include_dirs=["numba"])

    ext_modules = [ext_dynfunc, ext_dispatcher, ext_helperlib,
                   ext_typeconv, ext_np_ufunc, ext_npyufunc_num_threads,
                   ext_mviewbuf, ext_nrt_python, ext_jitclass_box,
                   ext_cuda_extras, ext_devicearray]

    ext_modules += ext_np_ufunc_backends

    return ext_modules


packages = find_packages(include=["numba", "numba.*"])

build_requires = ['numpy >={},<{}'.format(min_numpy_build_version,
                                          max_numpy_run_version)]
install_requires = [
    'llvmlite >={},<{}'.format(min_llvmlite_version, max_llvmlite_version),
    'numpy >={},<{}'.format(min_numpy_run_version, max_numpy_run_version),
    'setuptools<60',
    'importlib_metadata; python_version < "3.9"',
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
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Software Development :: Compilers",
    ],
    package_data={
        # HTML templates for type annotations
        "numba.core.annotations": ["*.html"],
        # Various test data
        "numba.cuda.tests.cudadrv.data": ["*.ptx", "*.cu"],
        "numba.cuda.tests.doc_examples.ffi": ["*.cu"],
        "numba.tests": ["pycc_distutils_usecase/*.py"],
        # Some C files are needed by pycc
        "numba": ["*.c", "*.h"],
        "numba.pycc": ["*.c", "*.h"],
        "numba.core.runtime": ["*.cpp", "*.c", "*.h"],
        "numba.cext": ["*.c", "*.h"],
        # numba gdb hook init command language file
        "numba.misc": ["cmdlang.gdb"],
        "numba.typed": ["py.typed"],
    },
    scripts=["numba/pycc/pycc", "bin/numba"],
    url="https://numba.pydata.org",
    packages=packages,
    setup_requires=build_requires,
    install_requires=install_requires,
    python_requires=">={}".format(min_python_version),
    license="BSD",
    cmdclass=cmdclass,
)

with open('README.rst') as f:
    metadata['long_description'] = f.read()

if is_building():
    metadata['ext_modules'] = get_ext_modules()

setup(**metadata)
