#! /usr/bin/env python
# ______________________________________________________________________
'''test_extern_call

Unit tests checking on Numba's code generation for Python/Numpy C-API calls.
'''
# ______________________________________________________________________

import numpy
from numba.decorators import numba_compile

import unittest

# ______________________________________________________________________

def call_zeros_like(arr):
    return numpy.zeros_like(arr)

# ______________________________________________________________________

class TestExternCall(unittest.TestCase):
    def test_call_zeros_like(self):
        testarr = numpy.array([1., 2, 3, 4, 5])
        testfn = numba_compile(arg_types = [['d']], ret_type = ['d'])(
            call_zeros_like)
        self.assertEqual(testfn(testarr), numpy.zeros_like(testarr))

# ______________________________________________________________________

if __name__ == "__main__":
    unittest.main()

# ______________________________________________________________________
# End of test_extern_call.py
