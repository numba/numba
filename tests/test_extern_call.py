#! /usr/bin/env python
# ______________________________________________________________________
'''test_extern_call

Unit tests checking on Numba's code generation for Python/Numpy C-API calls.
'''
# ______________________________________________________________________

from numba import *

import numpy
from numba.decorators import numba_compile

import unittest

# ______________________________________________________________________

def call_zeros_like(arr):
    return numpy.zeros_like(arr)

# ______________________________________________________________________

def call_len(arr):
    return len(arr)

# ______________________________________________________________________

class TestExternCall(unittest.TestCase):
    def test_call_zeros_like(self):
        testarr = numpy.array([1., 2, 3, 4, 5], dtype=numpy.double)
        testfn = numba_compile(arg_types = [double[:]], ret_type = double[:])(
            call_zeros_like)
        self.assertTrue((testfn(testarr) == numpy.zeros_like(testarr)).all())

    def test_call_len(self):
        testarr = numpy.arange(10.)
        testfn = numba_compile(arg_types = [double[:]], ret_type = long_)(
            call_len)
        self.assertEqual(testfn(testarr), 10)

# ______________________________________________________________________

if __name__ == "__main__":
    # TestExternCall('test_call_len').debug()
    unittest.main()

# ______________________________________________________________________
# End of test_extern_call.py
