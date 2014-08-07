from __future__ import print_function
import warnings
import numpy as np

import numba.unittest_support as unittest
from numba import jit
from numba.config import NumbaWarning


class TestBuiltins(unittest.TestCase):
    def test_type_infer_warning(self):
        def add(x, y):
            a = []
            return x + y

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always', NumbaWarning)

            cfunc = jit(add)
            cfunc(1, 2)

            self.assertEqual(len(w), 2)
            # Type inference failure
            self.assertEqual(w[0].category, NumbaWarning)
            self.assertIn('type inference', str(w[0].message))

            # Object mode
            self.assertEqual(w[1].category, NumbaWarning)
            self.assertIn('object mode', str(w[1].message))

    def test_return_type_warning(self):
        y = np.ones(4, dtype=np.float32)
        def return_external_array():
            return y

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always', NumbaWarning)

            cfunc = jit(return_external_array)
            cfunc()

            self.assertEqual(len(w), 2)
            # Legal return value failure
            self.assertEqual(w[0].category, NumbaWarning)
            self.assertIn('return type', str(w[0].message))

            # Object mode
            self.assertEqual(w[1].category, NumbaWarning)
            self.assertIn('object mode', str(w[1].message))

    def test_no_warning_with_forceobj(self):
        def add(x, y):
            a = []
            return x + y

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always', NumbaWarning)

            cfunc = jit(add, forceobj=True)
            cfunc(1, 2)

            self.assertEqual(len(w), 0)

if __name__ == '__main__':
    unittest.main()
