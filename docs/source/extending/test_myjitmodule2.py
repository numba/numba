import numpy as np
from numba import njit
from numba import unittest_support as unittest
from numba.tests import support
from numba.errors import TypingError

import mymodule
import myjitmodule2 # noqa - has side-effect, overload mymodule.set_to_x


@njit
def wrap_set_to_x(arr, x):
    mymodule.set_to_x(arr, x)


class TestSpam(support.TestCase):

    def test_int(self):
        a = np.arange(10)
        wrap_set_to_x(a, 1)
        self.assertPreciseEqual(np.ones(10, dtype=np.int64), a)

    def test_float(self):
        a = np.arange(10, dtype=np.float64)
        wrap_set_to_x(a, 1.0)
        self.assertPreciseEqual(np.ones(10), a)

    def test_float_exception_on_nan(self):
        a = np.arange(10, dtype=np.float64)
        a[0] = np.nan
        with self.assertRaises(ValueError) as e:
            wrap_set_to_x(a, 1.0)
        self.assertIn("no element of arr must be nan",
                      str(e.exception))

    def test_type_mismatch(self):
        a = np.arange(10)
        with self.assertRaises(TypingError) as e:
            wrap_set_to_x(a, 1.0)
        self.assertIn("the types of the input do not match",
                      str(e.exception))

    def test_exception_on_unsupported_dtype(self):
        a = np.arange(10, dtype=np.complex128)
        with self.assertRaises(TypingError) as e:
            wrap_set_to_x(a, np.complex128(1.0))
        self.assertIn("only integer and floating-point types allowed",
                      str(e.exception))

    def test_exception_on_tuple(self):
        a = (1, 2, 3)
        with self.assertRaises(TypingError) as e:
            wrap_set_to_x(a, 1)
        self.assertIn("tuple isn't allowed as input, use numpy arrays",
                      str(e.exception))


if __name__ == '__main__':
    unittest.main()
