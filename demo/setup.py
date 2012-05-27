from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import os

include_dirs = [os.path.join('..', 'include')]

extensions = [
    Extension("consumer", ["consumer.pyx", "consumer_c_code.c"],
              include_dirs=include_dirs),
    Extension("provider", ["provider.pyx"],
              include_dirs=include_dirs)]

setup(cmdclass={'build_ext': build_ext},
      ext_modules=extensions)
