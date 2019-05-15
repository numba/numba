from __future__ import print_function
import warnings
import numpy as np

import numba.unittest_support as unittest
from numba import jit
from numba.errors import NumbaWarning, deprecated, NumbaDeprecationWarning
from numba import errors


class TestBuiltins(unittest.TestCase):

    def check_objmode_deprecation_warning(self, w):
        # Object mode fall-back is slated for deprecation, check the warning
        msg = ("Fall-back from the nopython compilation path to the object "
               "mode compilation path has been detected")
        self.assertEqual(w.category, NumbaDeprecationWarning)
        self.assertIn(msg, str(w.message))

    def test_type_infer_warning(self):
        def add(x, y):
            a = {} # noqa dead
            return x + y

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always', NumbaWarning)

            cfunc = jit(add)
            cfunc(1, 2)

            self.assertEqual(len(w), 3)
            # Type inference failure
            self.assertEqual(w[0].category, NumbaWarning)
            self.assertIn('type inference', str(w[0].message))

            # Object mode
            self.assertEqual(w[1].category, NumbaWarning)
            self.assertIn('object mode', str(w[1].message))

            # check objmode deprecation warning
            self.check_objmode_deprecation_warning(w[2])

    def test_return_type_warning(self):
        y = np.ones(4, dtype=np.float32)

        def return_external_array():
            return y

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always', NumbaWarning)

            cfunc = jit(_nrt=False)(return_external_array)
            cfunc()

            self.assertEqual(len(w), 3)

            # Legal return value failure
            self.assertEqual(w[0].category, NumbaWarning)
            self.assertIn('return type', str(w[0].message))

            # Object mode fall-back
            self.assertEqual(w[1].category, NumbaWarning)
            self.assertIn('object mode without forceobj=True',
                          str(w[1].message))

            # check objmode deprecation warning
            self.check_objmode_deprecation_warning(w[2])

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
            a = [] # noqa dead
            return x + y

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always', NumbaWarning)

            cfunc = jit(add, forceobj=True)
            cfunc(1, 2)

            self.assertEqual(len(w), 0)

    def test_loop_lift_warn(self):
        def do_loop(x):
            a = {} # noqa dead
            for i in range(x.shape[0]):
                x[i] *= 2

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always', NumbaWarning)

            x = np.ones(4, dtype=np.float32)
            cfunc = jit(do_loop)
            cfunc(x)

            self.assertEqual(len(w), 4)

            # Type inference failure (1st pass, in npm, fall-back to objmode
            # with looplift)
            self.assertEqual(w[0].category, NumbaWarning)
            self.assertIn('type inference', str(w[0].message))
            self.assertIn('WITH looplifting', str(w[0].message))

            # Type inference failure (2nd pass, objmode with lifted loops,
            # loop found but still failed, fall back to objmode no looplift)
            self.assertEqual(w[1].category, NumbaWarning)
            self.assertIn('type inference', str(w[1].message))
            self.assertIn('WITHOUT looplifting', str(w[1].message))

            # States compilation outcome
            self.assertEqual(w[2].category, NumbaWarning)
            self.assertIn('compiled in object mode without forceobj=True',
                          str(w[2].message))
            self.assertIn('but has lifted loops', str(w[2].message))

            # check objmode deprecation warning
            self.check_objmode_deprecation_warning(w[3])

    def test_deprecated(self):
        @deprecated('foo')
        def bar():
            pass

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            bar()

            self.assertEqual(len(w), 1)
            self.assertEqual(w[0].category, DeprecationWarning)
            self.assertIn('bar', str(w[0].message))
            self.assertIn('foo', str(w[0].message))

    def test_warnings_fixer(self):
        # For some context, see #4083

        wfix = errors.WarningsFixer(errors.NumbaWarning)
        with wfix.catch_warnings('foo', 10):
            warnings.warn(errors.NumbaWarning('same'))
            warnings.warn(errors.NumbaDeprecationWarning('same'))

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            wfix.flush()

            self.assertEqual(len(w), 2)
            # the order of these will be backwards to the above, the
            # WarningsFixer flush method sorts with a key based on str
            # comparison
            self.assertEqual(w[0].category, NumbaDeprecationWarning)
            self.assertEqual(w[1].category, NumbaWarning)
            self.assertIn('same', str(w[0].message))
            self.assertIn('same', str(w[1].message))


if __name__ == '__main__':
    unittest.main()
