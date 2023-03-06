import warnings
from numba import jit, generated_jit
from numba.core.errors import (NumbaDeprecationWarning,
                               NumbaPendingDeprecationWarning, NumbaWarning)
from numba.tests.support import TestCase
import unittest


class TestDeprecation(TestCase):

    def check_warning(self, warnings, expected_str, category):
        self.assertEqual(len(warnings), 1)
        self.assertEqual(warnings[0].category, category)
        self.assertIn(expected_str, str(warnings[0].message))
        self.assertIn("https://numba.readthedocs.io", str(warnings[0].message))

    @TestCase.run_test_in_subprocess
    def test_jitfallback(self):
        # tests that @jit falling back to object mode raises a
        # NumbaDeprecationWarning
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("ignore", category=NumbaWarning)
            warnings.simplefilter("always", category=NumbaDeprecationWarning)
            # ignore the warning about the nopython kwarg not being supplied
            warnings.filterwarnings("ignore",
                                    message=(r".*The 'nopython' keyword "
                                             r"argument was not supplied.*"),
                                    category=NumbaDeprecationWarning,)

            def foo():
                return []  # empty list cannot be typed
            jit(foo)()

            msg = ("Fall-back from the nopython compilation path to the object "
                   "mode compilation path")

            self.check_warning(w, msg, NumbaDeprecationWarning)

    @TestCase.run_test_in_subprocess
    def test_default_missing_nopython_kwarg(self):
        # test that not supplying `nopython` kwarg to @jit raises a warning
        # about the default changing.
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("ignore", category=NumbaWarning)
            warnings.simplefilter("always", category=NumbaDeprecationWarning)

            @jit
            def foo():
                pass

            foo()

            msg = "The 'nopython' keyword argument was not supplied"
            self.check_warning(w, msg, NumbaDeprecationWarning)

    @TestCase.run_test_in_subprocess
    def test_explicit_false_nopython_kwarg(self):
        # tests that explicitly setting `nopython=False` in @jit raises a
        # warning about the default changing and it being an error in the
        # future.
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("ignore", category=NumbaWarning)
            warnings.simplefilter("always", category=NumbaDeprecationWarning)

            @jit(nopython=False)
            def foo():
                pass

            foo()

            msg = "The keyword argument 'nopython=False' was supplied"
            self.check_warning(w, msg, NumbaDeprecationWarning)

    @TestCase.run_test_in_subprocess
    def test_default_missing_nopython_kwarg_silent_if_forceobj(self):
        # Checks that if forceobj is set and the nopython kwarg is also not
        # present then no warning is raised. The user intentially wants objmode.

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("ignore", category=NumbaWarning)
            warnings.simplefilter("always", category=NumbaDeprecationWarning)

            @jit(forceobj=True)
            def foo():
                object()

            foo()

        # no warnings should be raised.
        self.assertFalse(w)

    @TestCase.run_test_in_subprocess
    def test_reflection_of_mutable_container(self):
        # tests that reflection in list/set warns
        def foo_list(a):
            return a.append(1)

        def foo_set(a):
            return a.add(1)

        for f in [foo_list, foo_set]:
            container = f.__name__.strip('foo_')
            inp = eval(container)([10, ])
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("ignore", category=NumbaWarning)
                warnings.simplefilter("always",
                                      category=NumbaPendingDeprecationWarning)
                jit(nopython=True)(f)(inp)
                self.assertEqual(len(w), 1)
                self.assertEqual(w[0].category, NumbaPendingDeprecationWarning)
                warn_msg = str(w[0].message)
                msg = ("Encountered the use of a type that is scheduled for "
                       "deprecation")
                self.assertIn(msg, warn_msg)
                msg = ("\'reflected %s\' found for argument" % container)
                self.assertIn(msg, warn_msg)
                self.assertIn("https://numba.readthedocs.io", warn_msg)

    @TestCase.run_test_in_subprocess
    def test_generated_jit(self):

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("ignore", category=NumbaWarning)
            warnings.simplefilter("always", category=NumbaDeprecationWarning)

            @generated_jit
            def bar():
                return lambda : None

            @jit(nopython=True)
            def foo():
                bar()

            foo()

            self.check_warning(w, "numba.generated_jit is deprecated",
                               NumbaDeprecationWarning)

    @TestCase.run_test_in_subprocess
    def test_pycc_module(self):
        # checks import of module warns

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always",
                                  category=NumbaPendingDeprecationWarning)
            import numba.pycc # noqa: F401

            expected_str = ("The 'pycc' module is pending deprecation.")
            self.check_warning(w, expected_str, NumbaPendingDeprecationWarning)

    @TestCase.run_test_in_subprocess
    def test_pycc_CC(self):
        # check the most commonly used functionality (CC) warns

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always",
                                  category=NumbaPendingDeprecationWarning)
            from numba.pycc import CC # noqa: F401

            expected_str = ("The 'pycc' module is pending deprecation.")
            self.check_warning(w, expected_str, NumbaPendingDeprecationWarning)


if __name__ == '__main__':
    unittest.main()
