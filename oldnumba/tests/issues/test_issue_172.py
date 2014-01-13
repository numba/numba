# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import

import numpy as np
import numba as nb
from numba import jit, int_, void, float64

@jit
class Foo(object):
    @void(int_)
    def __init__(self, size):
        self.arr = np.zeros(size, dtype=float)
        self.type = nb.typeof(self.arr)

assert Foo(10).type == float64[:]
