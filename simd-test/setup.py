from distutils.core import setup, Extension

import numpy
import platform
import os

cc_args = ['-std=c++11', '-O3', '-fstrict-aliasing', '-fomit-frame-pointer', '-mavx']
ld_args = []
if platform.system() == 'Darwin':
    ld_args += ['-framework', 'Accelerate']
    os.putenv('CC', 'clang')


module1 = Extension(name = 'simdtest',
                    sources=['simdtest.cpp', 'vector-machine.cpp'],
                    include_dirs = [numpy.get_include()],
                    extra_compile_args = cc_args,
                    extra_link_args = ld_args)

setup(name = 'simdtest',
      author = 'Continuum Analytics, Inc.',
      version='0.1',
      description='Testing a kernel using intel SIMD extensions',
      ext_modules = [module1])

