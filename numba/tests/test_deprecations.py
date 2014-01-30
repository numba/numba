from __future__ import print_function, absolute_import
import warnings
from numba import jit, autojit, vectorize
import numba.unittest_support as unittest


def dummy(): pass


def stub_vec(a):
    return a


class TestDeprecation(unittest.TestCase):

    def test_autojit(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            autojit(dummy)
            self.assertEqual(len(w), 1)


if __name__ == '__main__':
    unittest.main()

