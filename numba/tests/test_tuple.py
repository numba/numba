#! /usr/bin/env python
# ______________________________________________________________________
'''test_tuple

Unit test aimed at testing symbolic execution of the BUILD_TUPLE opcode.
'''
# ______________________________________________________________________

import numpy

from numba import *
from numba.decorators import jit
from numba.tests import test_support

import unittest

# ______________________________________________________________________

def tuple_fn_0 (inarr):
    i = 1
    j = 2
    k = 3
    internal_tuple = (i, j, k)
    return inarr[internal_tuple]
#    return inarr[1,2,3]

# ______________________________________________________________________

class TestTuple (test_support.ByteCodeTestCase):
    def test_tuple_fn_0 (self):
        test_arr = numpy.zeros((4,4,4))
        compiled_fn = self.jit(argtypes = [double[:,:,:]])(tuple_fn_0)
        self.assertEqual(compiled_fn(test_arr), 0.)

class TestASTTuple(test_support.ASTTestCase, TestTuple):
    pass

# ______________________________________________________________________

if __name__ == "__main__":
    unittest.main()

# ______________________________________________________________________
# End of test_tuple.py
