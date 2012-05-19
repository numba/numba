from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import os

include_dirs = [os.path.join('..', 'src')]

extensions = [
    Extension("consumer", ["consumer.pyx", "../src/extensibletype.c"],
              include_dirs=include_dirs),
    Extension("provider", ["provider.c", "../src/extensibletype.c"])]

setup(cmdclass={'build_ext': build_ext},
      ext_modules=extensions)

# Note setuptools messes things up, as usual, and a hack is needed to
# have it work together with Cython, see, e.g.,
# https://github.com/pydata/pandas/fake_pyrex
