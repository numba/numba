from distutils.core import setup, Extension

module1 = Extension('npufunc', sources=['logit.c'])

setup(name = 'npufunc',
        version='1.0',
        description='Parallel vector tryout',
        ext_modules = [module1])
