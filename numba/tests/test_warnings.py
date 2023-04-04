import os
import subprocess
import sys
import warnings
import numpy as np

import unittest
from numba import jit
from numba.core.errors import NumbaWarning, deprecated, NumbaDeprecationWarning
from numba.core import errors
from numba.tests.support import ignore_internal_warnings


class TestBuiltins(unittest.TestCase):

    def check_objmode_deprecation_warning(self, w):
        # Object mode fall-back is slated for deprecation, check the warning
        msg = ("Fall-back from the nopython compilation path to the object "
               "mode compilation path has been detected")
        self.assertEqual(w.category, NumbaDeprecationWarning)
        self.assertIn(msg, str(w.message))

    def check_nopython_kwarg_missing_warning(self, w):
        # nopython default is scheduled to change when objmode fall-back is
        # removed, check warning.
        msg = ("The \'nopython\' keyword argument was not supplied")
        self.assertEqual(w.category, NumbaDeprecationWarning)
        self.assertIn(msg, str(w.message))

    def test_type_infer_warning(self):
        def add(x, y):
            a = {} # noqa dead
            return x + y

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always', NumbaWarning)
            ignore_internal_warnings()

            cfunc = jit(add)
            cfunc(1, 2)

            self.assertEqual(len(w), 4)

            # 'nopython=' kwarg was not supplied to @jit
            self.check_nopython_kwarg_missing_warning(w[0])

            # Type inference failure
            self.assertEqual(w[1].category, NumbaWarning)
            self.assertIn('type inference', str(w[1].message))

            # Object mode
            self.assertEqual(w[2].category, NumbaWarning)
            self.assertIn('object mode', str(w[2].message))

            # check objmode deprecation warning
            self.check_objmode_deprecation_warning(w[3])

    def test_return_type_warning(self):
        y = np.ones(4, dtype=np.float32)

        def return_external_array():
            return y

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always', NumbaWarning)
            ignore_internal_warnings()

            cfunc = jit(_nrt=False)(return_external_array)
            cfunc()

            self.assertEqual(len(w), 4)

            # 'nopython=' kwarg was not supplied to @jit
            self.check_nopython_kwarg_missing_warning(w[0])

            # Legal return value failure
            self.assertEqual(w[1].category, NumbaWarning)
            self.assertIn('return type', str(w[1].message))

            # Object mode fall-back
            self.assertEqual(w[2].category, NumbaWarning)
            self.assertIn('object mode without forceobj=True',
                          str(w[2].message))

            # check objmode deprecation warning
            self.check_objmode_deprecation_warning(w[3])

    def test_return_type_warning_with_nrt(self):
        """
        Rerun test_return_type_warning with nrt
        """
        y = np.ones(4, dtype=np.float32)

        def return_external_array():
            return y

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always', NumbaWarning)
            ignore_internal_warnings()

            cfunc = jit(nopython=True)(return_external_array)
            cfunc()
            # No more warning
            self.assertEqual(len(w), 0)

    def test_no_warning_with_forceobj(self):
        def add(x, y):
            a = [] # noqa dead
            return x + y

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always', NumbaWarning)
            ignore_internal_warnings()

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
            ignore_internal_warnings()

            x = np.ones(4, dtype=np.float32)
            cfunc = jit(do_loop)
            cfunc(x)

            msg = '\n'.join(f"----------\n{x.message}" for x in w)

            self.assertEqual(len(w), 5, msg=msg)

            # 'nopython=' kwarg was not supplied to @jit
            self.check_nopython_kwarg_missing_warning(w[0])

            # Type inference failure (1st pass, in npm, fall-back to objmode
            # with looplift)
            self.assertEqual(w[1].category, NumbaWarning)
            self.assertIn('type inference', str(w[1].message))
            self.assertIn('WITH looplifting', str(w[1].message))

            # Type inference failure (2nd pass, objmode with lifted loops,
            # loop found but still failed, fall back to objmode no looplift)
            self.assertEqual(w[1].category, NumbaWarning)
            self.assertIn('type inference', str(w[2].message))
            self.assertIn('WITHOUT looplifting', str(w[2].message))

            # States compilation outcome
            self.assertEqual(w[3].category, NumbaWarning)
            self.assertIn('compiled in object mode without forceobj=True',
                          str(w[3].message))
            self.assertIn('but has lifted loops', str(w[3].message))

            # check objmode deprecation warning
            self.check_objmode_deprecation_warning(w[4])

    def test_deprecated(self):
        @deprecated('foo')
        def bar():
            pass

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            ignore_internal_warnings()
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
            ignore_internal_warnings()

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            ignore_internal_warnings()
            wfix.flush()

            self.assertEqual(len(w), 2)
            # the order of these will be backwards to the above, the
            # WarningsFixer flush method sorts with a key based on str
            # comparison
            self.assertEqual(w[0].category, NumbaDeprecationWarning)
            self.assertEqual(w[1].category, NumbaWarning)
            self.assertIn('same', str(w[0].message))
            self.assertIn('same', str(w[1].message))

    def test_disable_performance_warnings(self):

        not_found_ret_code = 55
        found_ret_code = 99
        expected = "'parallel=True' was specified but no transformation"

        # NOTE: the error_usecases is needed as the NumbaPerformanceWarning's
        # for parallel=True failing to parallelise do not appear for functions
        # defined by string eval/exec etc.
        parallel_code = """if 1:
            import warnings
            from numba.tests.error_usecases import foo
            import numba
            from numba.tests.support import ignore_internal_warnings
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter('always')
                ignore_internal_warnings()
                foo()
            for x in w:
                if x.category == numba.errors.NumbaPerformanceWarning:
                    if "%s" in str(x.message):
                        exit(%s)
            exit(%s)
        """ % (expected, found_ret_code, not_found_ret_code)

        # run in the standard env, warning should raise
        popen = subprocess.Popen([sys.executable, "-c", parallel_code])
        out, err = popen.communicate()
        self.assertEqual(popen.returncode, found_ret_code)

        # run in an env with performance warnings disabled, should not warn
        env = dict(os.environ)
        env['NUMBA_DISABLE_PERFORMANCE_WARNINGS'] = "1"
        popen = subprocess.Popen([sys.executable, "-c", parallel_code], env=env)
        out, err = popen.communicate()
        self.assertEqual(popen.returncode, not_found_ret_code)


if __name__ == '__main__':
    unittest.main()
