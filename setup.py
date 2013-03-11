from distutils.core import setup
from Cython.Distutils import build_ext

from setupconfig import get_extensions

setup(cmdclass={'build_ext': build_ext},
      ext_modules=get_extensions(prefix=""))
