# -*- coding: utf-8 -*-

from __future__ import print_function, division, absolute_import

from numba.typesystem import typesystem, universe
from numba.typesystem import numba_typesystem as ts, llvm_typesystem as lts

import llvm.core
pointer = llvm.core.Type.pointer

def convert(ts1, ts2, conversion_type, typenames):
    for typename in typenames:
        t1 = getattr(ts1, typename)
        t2 = getattr(ts2, typename)
        assert ts.convert(conversion_type, t1) == t2, (t1, t2)

def test_numeric_conversion():
    typenames = universe.int_typenames + universe.float_typenames
    convert(ts, lts, "llvm", typenames)


test_numeric_conversion()