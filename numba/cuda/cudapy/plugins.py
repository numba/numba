from __future__ import print_function, absolute_import, division
from functools import reduce
import operator
import warnings
from numba.cuda.compiler import CUDAKernel, CUDAKernelBase, AutoJitCUDAKernel
from numba.cuda.descriptor import CUDATargetDesc
from . import printimpl

# Extend target features

CUDATargetDesc.targetctx.insert_func_defn(printimpl.registry.functions)


