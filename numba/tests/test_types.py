#! /usr/bin/env python
# ______________________________________________________________________
'''
Test type mapping.
'''
# ______________________________________________________________________

import numba
from numba import *
from numba.decorators import jit
from numba.tests import test_support

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

class TestIf(test_support.ByteCodeTestCase):
    def test_int(self):
        func = self.jit(restype=numba.int_,
                                 argtypes=[numba.int_])(test_int)
        self.assertEqual(func(-1), 42)
        self.assertEqual(func(1), 22)

    def test_long(self):
        func = self.jit(restype=numba.long_,
                                 argtypes=[numba.long_])(test_long)
        self.assertEqual(func(-1), 42)
        self.assertEqual(func(1), 22)
# ______________________________________________________________________

class TestASTIf(test_support.ASTTestCase, TestIf):
    pass

if __name__ == "__main__":
    unittest.main()

# ______________________________________________________________________
# End of test_if.py

