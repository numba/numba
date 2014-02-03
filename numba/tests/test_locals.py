from __future__ import print_function, division, absolute_import
from numba import jit, float32
from numba import unittest_support as unittest

def foo():
    x = 123
    return x


class TestLocals(unittest.TestCase):
    def test_seed_types(self):
        cfoo = jit((), locals={'x': float32})(foo)
        cres = list(cfoo.overloads.values())[0]
        self.assertEqual(cres.signature.return_type, float32)


if __name__ == '__main__':
    unittest.main()
