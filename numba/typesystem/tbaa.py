# -*- coding: utf-8 -*-

"""
Some types to aid in type-based alias analysis. See numba/metadata.py.
"""

from __future__ import print_function, division, absolute_import

from numba.typesystem.types import NumbaType
from numba.typesystem import object_, npy_intp

class TBAAType(NumbaType):
    is_tbaa = True
    typename = "tbaa_type"
    argnames = ["name", "root"]

numpy_array = TBAAType("numpy array", object_)
numpy_shape = TBAAType("numpy shape", npy_intp.pointer())
numpy_strides = TBAAType("numpy strides", npy_intp.pointer())
numpy_ndim = TBAAType("numpy flags", npy_intp.pointer())
numpy_dtype = TBAAType("numpy dtype", object_)
numpy_base = TBAAType("numpy base", object_)
numpy_flags = TBAAType("numpy flags", npy_intp.pointer())
