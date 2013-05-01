# -*- coding: utf-8 -*-

"""
Some types to aid in type-based alias analysis.
"""

from __future__ import print_function, division, absolute_import

from numba.typesystem import *

class TBAAType(NumbaType):
    """
    Type based alias analysis type. See numba/metadata.py.
    """

    is_tbaa = True

    def __init__(self, name, root, **kwds):
        super(TBAAType, self).__init__(**kwds)
        self.name = name
        self.root = root

numpy_array = TBAAType("numpy array", root=object_)
numpy_shape = TBAAType("numpy shape", root=intp.pointer())
numpy_strides = TBAAType("numpy strides", root=intp.pointer())
numpy_ndim = TBAAType("numpy flags", root=int_.pointer())
numpy_dtype = TBAAType("numpy dtype", root=object_)
numpy_base = TBAAType("numpy base", root=object_)
numpy_flags = TBAAType("numpy flags", root=int_.pointer())
