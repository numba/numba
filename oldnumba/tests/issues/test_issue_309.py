# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import
import numba

@numba.jit(numba.float32(numba.float32))
def trial0(x):
    if x > 0.0:
        return x * 3.0 + 5.0
    else:
        raise IndexError

try:
    trial0(-1.0)
except IndexError:
    pass
else:
    raise Exception("expected indexerror")

assert trial0(2.0) == 11.0
