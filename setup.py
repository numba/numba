from distutils.core import setup, Extension
import os
import numpy
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

ext_dynfunc = Extension(name='numba._dynfunc', sources=['numba/_dynfunc.c'],
                        extra_compile_args=CFLAGS)

ext_numpyadapt = Extension(name='numba._numpyadapt',
                           sources=['numba/_numpyadapt.c'],
                           include_dirs=[numpy.get_include()],
                           extra_compile_args=CFLAGS)

ext_dispatcher = Extension(name="numba._dispatcher",
                           include_dirs=[numpy.get_include()],
                           sources=['numba/_dispatcher.c',
                                    'numba/_dispatcherimpl.cpp',
                                    'numba/typeconv/typeconv.cpp'])

ext_helperlib = Extension(name="numba._helperlib",
                          sources=["numba/_helperlib.c", "numba/_math_c99.c"],
                          extra_compile_args=CFLAGS)

ext_typeconv = Extension(name="numba.typeconv._typeconv",
                         sources=["numba/typeconv/typeconv.cpp",
                                  "numba/typeconv/_typeconv.cpp"])

ext_npyufunc_ufunc = Extension(name="numba.npyufunc._internal",
                               sources=["numba/npyufunc/_internal.c"],
                               include_dirs=[numpy.get_include()],
                               depends=["numba/npyufunc/_internal.h"])


ext_modules = [ext_dynfunc, ext_numpyadapt, ext_dispatcher, ext_helperlib,
               ext_typeconv, ext_npyufunc_ufunc]

packages = [
    "numba",
    "numba.targets",
    "numba.tests",
    "numba.typing",
    "numba.typeconv",
    "numba.npyufunc",
    "numba.pycc",
]

setup(name='numba',
      description="compiling Python code using LLVM",
      version=versioneer.get_version(),

      classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        # "Programming Language :: Python :: 2.6",
        "Programming Language :: Python :: 2.7",
        # "Programming Language :: Python :: 3.3",
        "Topic :: Utilities",
      ],
      scripts=["numba/pycc/pycc"],
      author="Continuum Analytics, Inc.",
      author_email="numba-users@continuum.io",
      url="http://numba.github.com",
      ext_modules=ext_modules,
      packages=packages,
      license="BSD",
      cmdclass=cmdclass,
      **setup_args)
