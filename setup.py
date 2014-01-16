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
                           sources=['numba/_dispatcher.c',],
                           extra_compile_args=CFLAGS)

ext_helperlib = Extension(name="numba._helperlib",
                          sources=["numba/_helperlib.c"],
                          extra_compile_args=CFLAGS)

ext_modules = [ext_dynfunc, ext_numpyadapt, ext_dispatcher, ext_helperlib]

packages = [
    "numba",
    "numba.targets",
    "numba.tests",
    "numba.typing",
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
        # "Programming Language :: Python :: 3.2",
        "Topic :: Utilities",
      ],
      author="Continuum Analytics, Inc.",
      author_email="numba-users@continuum.io",
      ext_modules=ext_modules,
      packages=packages,
      license="BSD",
      cmdclass=cmdclass,
      **setup_args)
