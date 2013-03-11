from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import os

import numpy as np

include_dirs = ['include', '../ulib/src/base', np.get_include()]

perfecthash_deps = ["include/perfecthash.h"]

extensions = [
    Extension("extensibletype.extensibletype",
              [os.path.join("extensibletype", "extensibletype.pyx"),
               #'../ulib/src/base/md5sum.c',
               #'../ulib/src/base/hash.c'
              ],
              include_dirs=include_dirs,
              depends=perfecthash_deps),
    Extension("extensibletype.intern",
              ["extensibletype/intern.pyx"],
              include_dirs=include_dirs,
              depends=["include/globalinterning.h",
                       "include/interning.h",
                       "include/perfecthash.h"]),
    Extension("extensibletype.methodtable",
              [os.path.join("extensibletype", "methodtable.pyx")],
              include_dirs=include_dirs,
              depends=perfecthash_deps),
]

setup(cmdclass={'build_ext': build_ext},
      ext_modules=extensions)
