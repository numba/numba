from __future__ import print_function, division, absolute_import

from numba import compiler, float32
from numba import unittest_support as unittest

def foo():
    x = 123
    return x


class TestLocals(unittest.TestCase):

    def test_seed_types(self):
        cres = compiler.compile_isolated(foo, (), locals={'x': float32})
        self.assertEqual(cres.signature.return_type, float32)


if __name__ == '__main__':
    unittest.main()
