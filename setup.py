try:
    # Try to use setuptools so as to enable support of the special
    # "Microsoft Visual C++ Compiler for Python 2.7" (http://aka.ms/vcpython27)
    # for building under Windows.
    # Note setuptools >= 6.0 is required for this.
    from setuptools import setup, Extension
except ImportError:
    from distutils.core import setup, Extension

import sys
import os
import numpy
import numpy.distutils.misc_util as np_misc
import versioneer

versioneer.versionfile_source = 'numba/_version.py'
versioneer.versionfile_build = 'numba/_version.py'
versioneer.tag_prefix = ''
versioneer.parentdir_prefix = 'numba-'

cmdclass = versioneer.get_cmdclass()

setup_args = {
    'long_description': open('README.md').read(),
}

GCCFLAGS = ["-std=c89", "-Wdeclaration-after-statement", "-Werror"]

if os.environ.get("NUMBA_GCC_FLAGS"):
    CFLAGS = GCCFLAGS
else:
    CFLAGS = []


if sys.platform == 'darwin' and sys.version_info[:2] == (2, 6):
    cpp_link_args = ['-lstdc++']
else:
    cpp_link_args = []

npymath_info = np_misc.get_info('npymath')

ext_dynfunc = Extension(name='numba._dynfunc', sources=['numba/_dynfunc.c'],
                        extra_compile_args=CFLAGS,
                        depends=["numba/_pymodule.h"])

ext_npymath_exports = Extension(name='numba._npymath_exports',
                                sources=['numba/_npymath_exports.c'],
                                include_dirs=npymath_info['include_dirs'],
                                libraries=npymath_info['libraries'],
                                library_dirs=npymath_info['library_dirs'],
                                define_macros=npymath_info['define_macros'])


ext_dispatcher = Extension(name="numba._dispatcher",
                           include_dirs=[numpy.get_include()],
                           sources=['numba/_dispatcher.c',
                                    'numba/_dispatcherimpl.cpp',
                                    'numba/typeconv/typeconv.cpp'],
                           depends=["numba/_pymodule.h",
                                    "numba/_dispatcher.h"],
                           extra_link_args=cpp_link_args)

ext_helperlib = Extension(name="numba._helperlib",
                          include_dirs=[numpy.get_include()],
                          sources=["numba/_helperlib.c", "numba/_math_c99.c"],
                          extra_compile_args=CFLAGS,
                          depends=["numba/_pymodule.h",
                                   "numba/_math_c99.h",
                                   "numba/mathnames.inc"])

ext_typeconv = Extension(name="numba.typeconv._typeconv",
                         sources=["numba/typeconv/typeconv.cpp",
                                  "numba/typeconv/_typeconv.cpp"],
                         depends=["numba/_pymodule.h"],
                         extra_link_args=cpp_link_args)

ext_npyufunc_ufunc = Extension(name="numba.npyufunc._internal",
                               sources=["numba/npyufunc/_internal.c"],
                               include_dirs=[numpy.get_include()],
                               depends=["numba/npyufunc/_internal.h",
                                        "numba/_pymodule.h"])

ext_mviewbuf = Extension(name='numba.mviewbuf',
                         sources=['numba/mviewbuf.c'])


ext_modules = [ext_dynfunc, ext_npymath_exports, ext_dispatcher,

               ext_helperlib, ext_typeconv, ext_npyufunc_ufunc, ext_mviewbuf]


HSA_PATH = os.getenv("HSA_PATH")
if HSA_PATH is not None:
    # TODO: merge this two files into something in the numba repo
    hsa_support_srcs = ['numba/hsa/hsadrv/elf_utils.c']
    hsa_support_include = [os.path.join(HSA_PATH, 'include')]
    hsa_support_lib = [os.path.join(HSA_PATH, 'lib')]
    ext_hsa_support = Extension(name='numba.hsa.hsadrv._hsa_support',
                                sources=hsa_support_srcs,
                                include_dirs=hsa_support_include,
                                extra_compile_args=['-std=c99'],
                                extra_link_args=['-lelf'])
    ext_modules.append(ext_hsa_support)


packages = [
    "numba",
    "numba.targets",
    "numba.tests",
    "numba.typing",
    "numba.typeconv",
    "numba.npyufunc",
    "numba.pycc",
    "numba.servicelib",
    "numba.cuda",
    "numba.cuda.cudadrv",
    "numba.cuda.tests",
    "numba.cuda.tests.cudadrv",
    "numba.cuda.tests.cudadrv.data",
    "numba.cuda.tests.cudapy",
    "numba.hsa",
    "numba.hsa.hsadrv",
]

setup(name='numba',
      description="compiling Python code using LLVM",
      version=versioneer.get_version(),

      classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Programming Language :: Python :: 2.6",
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3.3",
        "Programming Language :: Python :: 3.4",
        "Topic :: Software Development :: Compilers",
      ],
      package_data={
        "numba": ["*.c", "*.h", "*.cpp", "*.inc"],
        "numba.npyufunc": ["*.c", "*.h"],
        "numba.typeconv": ["*.cpp", "*.hpp"],
        "numba.cuda.tests.cudadrv.data": ["*.ptx"],
      },
      scripts=["numba/pycc/pycc", "bin/numba"],
      author="Continuum Analytics, Inc.",
      author_email="numba-users@continuum.io",
      url="http://numba.github.com",
      ext_modules=ext_modules,
      packages=packages,
      license="BSD",
      cmdclass=cmdclass,
      **setup_args)

