# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import
from .decorators import Vectorize, GUVectorize, vectorize, guvectorize
from ._internal import PyUFunc_None, PyUFunc_Zero, PyUFunc_One
from . import _internal, array_exprs
if hasattr(_internal, 'PyUFunc_ReorderableNone'):
    PyUFunc_ReorderableNone = _internal.PyUFunc_ReorderableNone
del _internal, array_exprs
