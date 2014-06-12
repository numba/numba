from __future__ import print_function
import numba.unittest_support as unittest
from numba import jit
import sys


class TestFuncInterface(unittest.TestCase):
    def test_jit_function_docstring(self):

        def add(x, y):
            '''Return sum of two numbers'''
            return x + y

        c_add = jit(add)
        self.assertEqual(c_add.__doc__, 'Return sum of two numbers')

    def test_jit_function_code_object(self):
        def add(x, y):
            return x + y

        c_add = jit(add)
        if sys.version_info[0] >= 3:
            self.assertEqual(c_add.__code__, add.__code__)
        else:
            self.assertEqual(c_add.func_code, add.func_code)


if __name__ == '__main__':
    unittest.main()
