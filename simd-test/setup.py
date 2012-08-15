from distutils.core import setup, Extension

import numpy

module1 = Extension(name = 'simdtest',
                    sources=['simdtest.c'],
                    include_dirs = [numpy.get_include()],
                    extra_compile_args = ['-std=c99', '-march=core-avx-i', '-O3'])

setup(name = 'simdtest',
      author = 'Continuum Analytics, Inc.',
      version='0.1',
      description='Testing a kernel using intel SIMD extensions',
      ext_modules = [module1])
