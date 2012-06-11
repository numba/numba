from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import os

include_dirs = ['include']

extensions = [
    Extension("extensibletype.extensibletype",
              [os.path.join("extensibletype", "extensibletype.pyx")],
              include_dirs=include_dirs)]

setup(cmdclass={'build_ext': build_ext},
      ext_modules=extensions)
