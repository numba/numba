#! /usr/bin/env python
# ______________________________________________________________________
'''
Test type mapping.
'''
# ______________________________________________________________________

import numba
from numba import *
from numba.decorators import jit

import unittest

# ______________________________________________________________________

def test_int(arg):
    if arg > 0:
        result = 22
    else:
        result = 42
    return result

def test_long(arg):
    if arg > 0:
        result = 22
    else:
        result = 42
    return result

# ______________________________________________________________________

class TestIf(unittest.TestCase):
    def test_int(self):
        func = jit(ret_type=numba.int_,
                                 arg_types=[numba.int_])(test_int)
        self.assertEqual(func(-1), 42)
        self.assertEqual(func(1), 22)

    def test_long(self):
        func = jit(ret_type=numba.long_,
                                 arg_types=[numba.long_])(test_long)
        self.assertEqual(func(-1), 42)
        self.assertEqual(func(1), 22)
# ______________________________________________________________________

if __name__ == "__main__":
    unittest.main()

# ______________________________________________________________________
# End of test_if.py

