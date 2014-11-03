from numba import unittest_support as unittest
from numba import njit, stack_array, types
import numpy as np

@njit
def declaration_only():
    a = stack_array.new(3, types.float64)

@njit
def decl_setitem():
    a = stack_array.new(3, types.float64)
    a[0] = 10.0
    a[1] = 20.0
    a[2] = 30.0

@njit
def decl_setitem_getitem():
    a = stack_array.new(3, types.float64)
    a[0] = 10.0
    a[1] = 20.0
    a[2] = 30.0
    s = 0
    for i in range(3):
        s += a[i]
    return s

class TestStackArrays(unittest.TestCase):
    def test_declaration(self):
        '''
        Tests that no error is thrown in the generation of code for a stack
        array declaration.
        '''
        declaration_only()

    def test_setitem(self):
        '''
        Tests that no error is thrown in the generation of code for setting
        items in a stack array.
        '''
        decl_setitem()

    def test_getitem(self):
        '''
        Provides a test of the operation of stack array declaration as well as
        setting and getting items.
        '''
        s = decl_setitem_getitem()
        np.testing.assert_allclose(60.0, s)

if __name__ == '__main__':
    unittest.main()
