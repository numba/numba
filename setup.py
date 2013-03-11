import os
from distutils.core import setup
from Cython.Distutils import build_ext

from setupconfig import get_extensions

root = os.path.dirname(os.path.abspath(__file__))
setup(cmdclass={'build_ext': build_ext},
      ext_modules=get_extensions(path_prefix=root))
