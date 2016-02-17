from __future__ import print_function
import warnings
import numpy as np

import numba.unittest_support as unittest
from numba import jit
from numba.errors import NumbaWarning


class TestBuiltins(unittest.TestCase):
    def test_type_infer_warning(self):
        def add(x, y):
            a = {}
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

            cfunc = jit(_nrt=False)(return_external_array)
            cfunc()

            self.assertEqual(len(w), 2)
            # Legal return value failure
            self.assertEqual(w[0].category, NumbaWarning)
            self.assertIn('return type', str(w[0].message))

            # Object mode
            self.assertEqual(w[1].category, NumbaWarning)
            self.assertIn('object mode', str(w[1].message))

    def test_return_type_warning_with_nrt(self):
        """
        Rerun test_return_type_warning with nrt
        """
        y = np.ones(4, dtype=np.float32)

        def return_external_array():
            return y

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always', NumbaWarning)

            cfunc = jit(return_external_array)
            cfunc()
            # No more warning
            self.assertEqual(len(w), 0)


    def test_no_warning_with_forceobj(self):
        def add(x, y):
            a = []
            return x + y

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always', NumbaWarning)

            cfunc = jit(add, forceobj=True)
            cfunc(1, 2)

            self.assertEqual(len(w), 0)

    def test_loop_lift_warn(self):
        def do_loop(x):
            a = {}
            for i in range(x.shape[0]):
                x[i] *= 2

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always', NumbaWarning)

            x = np.ones(4, dtype=np.float32)
            cfunc = jit(do_loop)
            cfunc(x)

            self.assertEqual(len(w), 3)
            # Type inference failure (1st pass)
            self.assertEqual(w[0].category, NumbaWarning)
            self.assertIn('type inference', str(w[0].message))

            # Type inference failure (2nd pass, with lifted loops)
            self.assertEqual(w[1].category, NumbaWarning)
            self.assertIn('type inference', str(w[1].message))

            # Object mode
            self.assertEqual(w[2].category, NumbaWarning)
            self.assertIn('object mode', str(w[2].message))
            self.assertIn('lifted loops', str(w[2].message))


if __name__ == '__main__':
    unittest.main()
