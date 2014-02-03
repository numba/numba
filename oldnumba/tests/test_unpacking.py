# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import
import ctypes
import unittest

import numba as nb
from numba import jit, list_, tuple_, object_, int_, sized_pointer, npy_intp

import numpy as np

A = np.empty((5, 6))
shape_t = sized_pointer(npy_intp, 2)

# ______________________________________________________________________

def unpack(x):
    x, y = x
    return x * y

# ______________________________________________________________________

class Iterable(object):
    """I don't work yet"""

    def __iter__(self):
        return iter((5, 6))

class Sequence(object):
    """I work"""

    def __getitem__(self, idx):
        return [5, 6][idx]

# ______________________________________________________________________

class TestUnpacking(unittest.TestCase):

    def test_unpacking(self):
        lunpack  = jit(int_(list_(int_, 2)))(unpack)
        tunpack  = jit(int_(tuple_(int_, 2)))(unpack)
        tounpack = jit(int_(tuple_(object_, 2)))(unpack)
        iunpack  = jit(int_(object_))(unpack)
        sunpack  = jit(int_(object_))(unpack)
        punpack  = jit(int_(shape_t), wrap=False)(unpack)

        self.assertEqual(lunpack([5, 6]), 30)
        self.assertEqual(tunpack((5, 6)), 30)
        self.assertEqual(tounpack((5, 6)), 30)
        # self.assertEqual(iunpack(Iterable()), 30)
        self.assertEqual(sunpack(Sequence()), 30)

        c_punpack = nb.addressof(punpack)
        self.assertEqual(c_punpack(A.ctypes.shape), 30)

if __name__ == "__main__":
    unittest.main()
