# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import

from .decorators import Vectorize, GUVectorize, vectorize, guvectorize
from ._internal import PyUFunc_None, PyUFunc_Zero, PyUFunc_One
from . import _internal, array_exprs
if hasattr(_internal, 'PyUFunc_ReorderableNone'):
    PyUFunc_ReorderableNone = _internal.PyUFunc_ReorderableNone
del _internal, array_exprs


def _init():

    def init_vectorize():
        from numba.cuda.vectorizers import CUDAVectorize
        return CUDAVectorize

    def init_guvectorize():
        from numba.cuda.vectorizers import CUDAGUFuncVectorize
        return CUDAGUFuncVectorize

    Vectorize.target_registry.ondemand['cuda'] = init_vectorize
    GUVectorize.target_registry.ondemand['cuda'] = init_guvectorize

_init()
del _init
