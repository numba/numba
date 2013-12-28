from distutils.core import setup, Extension

ext_modules = [ Extension(name='numba._dynfunc',
                          sources=['numba/_dynfunc.c']) ]

setup(name = 'numba',
      description = '',
      author = 'Siu Kwan Lam',
      ext_modules = ext_modules,
      license = "BSD")
