#! /usr/bin/env python
# ______________________________________________________________________
'''test_tuple

Unit test aimed at testing symbolic execution of the BUILD_TUPLE opcode.
'''
# ______________________________________________________________________

import numpy

from numba.decorators import jit

import unittest

# ______________________________________________________________________

def tuple_fn_0 (inarr):
    i = 1
    j = 2
    k = 3
    internal_tuple = (i, j, k)
    return inarr[internal_tuple]

# ______________________________________________________________________

class TestTuple (unittest.TestCase):
    def test_tuple_fn_0 (self):
        test_arr = numpy.zeros((4,4,4))
        compiled_fn = jit(arg_types = [['d']])(tuple_fn_0)
        self.assertEqual(compiled_fn(test_arr), 0.)

# ______________________________________________________________________

if __name__ == "__main__":
    unittest.main()

# ______________________________________________________________________
# End of test_tuple.py
