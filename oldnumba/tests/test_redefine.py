from numba import *
import unittest

class TestRedefine(unittest.TestCase):
    def test_redefine(self):

        def foo(x):
            return x + 1

        jfoo = jit(int32(int32))(foo)

        # Test original function
        self.assertTrue(jfoo(1), 2)


        jfoo = jit(int32(int32))(foo)

        # Test re-compiliation
        self.assertTrue(jfoo(2), 3)

        def foo(x):
            return x + 2

        jfoo = jit(int32(int32))(foo)

        # Test redefinition
        self.assertTrue(jfoo(1), 3)
