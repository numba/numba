from distutils.core import setup, Extension
import numpy

ext_dynfunc = Extension(name='numba._dynfunc', sources=['numba/_dynfunc.c'])

ext_numpyadapt = Extension(name='numba._numpyadapt',
                           sources=['numba/_numpyadapt.c'],
                           include_dirs=[numpy.get_include()])

ext_modules = [ext_dynfunc, ext_numpyadapt]

setup(name = 'numba',
      description = '',
      author = 'Siu Kwan Lam',
      ext_modules = ext_modules,
      license = "BSD")
