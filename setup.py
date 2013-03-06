from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import os

import numpy as np

include_dirs = ['include', '../ulib/src/base', np.get_include()]

extensions = [
    Extension("extensibletype.extensibletype",
              [os.path.join("extensibletype", "extensibletype.pyx"),
               #'../ulib/src/base/md5sum.c',
               #'../ulib/src/base/hash.c'
              ],
              include_dirs=include_dirs)]

setup(cmdclass={'build_ext': build_ext},
      ext_modules=extensions)
