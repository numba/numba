# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import

from .decorators import Vectorize, GUVectorize, vectorize, guvectorize
from ._internal import PyUFunc_None, PyUFunc_Zero, PyUFunc_One
from . import _internal, array_exprs, parfor
from .parallel import threading_layer
if hasattr(_internal, 'PyUFunc_ReorderableNone'):
    PyUFunc_ReorderableNone = _internal.PyUFunc_ReorderableNone
del _internal, array_exprs


def _init():

    def init_cuda_vectorize():
        from numba.cuda.vectorizers import CUDAVectorize
        return CUDAVectorize

    def init_cuda_guvectorize():
        from numba.cuda.vectorizers import CUDAGUFuncVectorize
        return CUDAGUFuncVectorize

    Vectorize.target_registry.ondemand['cuda'] = init_cuda_vectorize
    GUVectorize.target_registry.ondemand['cuda'] = init_cuda_guvectorize

    def init_roc_vectorize():
        from numba.roc.vectorizers import HsaVectorize
        return HsaVectorize

    def init_roc_guvectorize():
        from numba.roc.vectorizers import HsaGUFuncVectorize
        return HsaGUFuncVectorize

    Vectorize.target_registry.ondemand['roc'] = init_roc_vectorize
    GUVectorize.target_registry.ondemand['roc'] = init_roc_guvectorize

_init()
del _init
