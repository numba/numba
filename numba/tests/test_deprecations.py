import warnings
import unittest
from contextlib import contextmanager

from numba import jit, generated_jit, vectorize, guvectorize
from numba.core.errors import (NumbaDeprecationWarning,
                               NumbaPendingDeprecationWarning, NumbaWarning)
from numba.tests.support import TestCase, needs_setuptools


@contextmanager
def _catch_numba_deprecation_warnings():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("ignore", category=NumbaWarning)
        warnings.simplefilter("always", category=NumbaDeprecationWarning)
        yield w


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
        with _catch_numba_deprecation_warnings() as w:
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
        with _catch_numba_deprecation_warnings() as w:

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
        with _catch_numba_deprecation_warnings() as w:

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

        with _catch_numba_deprecation_warnings() as w:

            @jit(forceobj=True)
            def foo():
                object()

            foo()

        # no warnings should be raised.
        self.assertFalse(w)

    @TestCase.run_test_in_subprocess
    def test_vectorize_missing_nopython_kwarg_not_reported(self):
        # Checks that use of @vectorize without a nopython kwarg doesn't raise
        # a warning about lack of said kwarg.

        with _catch_numba_deprecation_warnings() as w:
            # This compiles via nopython mode directly
            @vectorize('float64(float64)')
            def foo(a):
                return a + 1

        self.assertFalse(w)

    @TestCase.run_test_in_subprocess
    def test_vectorize_nopython_false_is_reported(self):
        # Checks that use of @vectorize with nopython=False raises a warning
        # about supplying it.

        with _catch_numba_deprecation_warnings() as w:
            # This compiles via nopython mode directly
            @vectorize('float64(float64)', nopython=False)
            def foo(a):
                return a + 1

        msg = "The keyword argument 'nopython=False' was supplied"
        self.check_warning(w, msg, NumbaDeprecationWarning)

    @TestCase.run_test_in_subprocess
    def test_vectorize_objmode_missing_nopython_kwarg_not_reported(self):
        # Checks that use of @vectorize without a nopython kwarg doesn't raise
        # a warning about lack of the nopython kwarg, but does raise a warning
        # about falling back to obj mode to compile.

        with _catch_numba_deprecation_warnings() as w:
            # This compiles via objmode fallback
            @vectorize('float64(float64)')
            def foo(a):
                object()
                return a + 1

        msg = ("Fall-back from the nopython compilation path to the object "
               "mode compilation path")
        self.check_warning(w, msg, NumbaDeprecationWarning)

    @TestCase.run_test_in_subprocess
    def test_vectorize_objmode_nopython_false_is_reported(self):
        # Checks that use of @vectorize with a nopython kwarg set to False
        # doesn't raise a warning about lack of the nopython kwarg, but does
        # raise a warning about it being False and then a warning about falling
        # back to obj mode to compile.

        with _catch_numba_deprecation_warnings() as w:
            # This compiles via objmode fallback with a warning about
            # nopython=False supplied.
            @vectorize('float64(float64)', nopython=False)
            def foo(a):
                object()
                return a + 1

        # 2 warnings: 1. use of nopython=False, then, 2. objmode fallback
        self.assertEqual(len(w), 2)

        msg = "The keyword argument 'nopython=False' was supplied"
        self.check_warning([w[0]], msg, NumbaDeprecationWarning)

        msg = ("Fall-back from the nopython compilation path to the object "
               "mode compilation path")
        self.check_warning([w[1]], msg, NumbaDeprecationWarning)

    @TestCase.run_test_in_subprocess
    def test_vectorize_objmode_direct_compilation_no_warnings(self):
        # Checks that use of @vectorize with forceobj=True raises no warnings.

        with _catch_numba_deprecation_warnings() as w:
            # Compiles via objmode directly with no warnings raised
            @vectorize('float64(float64)', forceobj=True)
            def foo(a):
                object()
                return a + 1

        self.assertFalse(w)

    @TestCase.run_test_in_subprocess
    def test_vectorize_objmode_compilation_nopython_false_no_warnings(self):
        # Checks that use of @vectorize with forceobj set and nopython set as
        # False raises no warnings.

        with _catch_numba_deprecation_warnings() as w:
            # Compiles via objmode directly with no warnings raised
            @vectorize('float64(float64)', forceobj=True, nopython=False)
            def foo(a):
                object()
                return a + 1

        self.assertFalse(w)

    @TestCase.run_test_in_subprocess
    def test_vectorize_parallel_true_no_warnings(self):
        # Checks that use of @vectorize with the parallel target doesn't
        # raise warnings about nopython kwarg, the parallel target doesn't
        # support objmode so nopython=True is implicit.
        with _catch_numba_deprecation_warnings() as w:
            @vectorize('float64(float64)', target='parallel')
            def foo(x):
                return x + 1

        self.assertFalse(w)

    @TestCase.run_test_in_subprocess
    def test_vectorize_parallel_true_nopython_true_no_warnings(self):
        # Checks that use of @vectorize with the parallel target and
        # nopython=True doesn't raise warnings about nopython kwarg.
        with _catch_numba_deprecation_warnings() as w:
            @vectorize('float64(float64)', target='parallel', nopython=True)
            def foo(x):
                return x + 1

        self.assertFalse(w)

    @TestCase.run_test_in_subprocess
    def test_vectorize_parallel_true_nopython_false_warns(self):
        # Checks that use of @vectorize with the parallel target and
        # nopython=False raises a warning about the nopython kwarg being False.
        with _catch_numba_deprecation_warnings() as w:
            @vectorize('float64(float64)', target='parallel', nopython=False)
            def foo(x):
                return x + 1

        msg = "The keyword argument 'nopython=False' was supplied"
        self.check_warning(w, msg, NumbaDeprecationWarning)

    @TestCase.run_test_in_subprocess
    def test_vectorize_calling_jit_with_nopython_false_warns_from_jit(self):
        # Checks the scope of the suppression of deprecation warnings that are
        # present in e.g. vectorize. The function `bar` should raise a
        # deprecation warning, the `@vectorize`d `foo` function should not,
        # even though both don't have a nopython kwarg.

        # First check that the @vectorize call doesn't raise anything
        with _catch_numba_deprecation_warnings() as w:
            @vectorize('float64(float64)', forceobj=True)
            def foo(x):
                return bar(x + 1)

            def bar(*args):
                pass

        self.assertFalse(w)

        # Now check the same compilation but this time compiling through to a
        # @jit call.
        with _catch_numba_deprecation_warnings() as w:
            @vectorize('float64(float64)', forceobj=True)
            def foo(x): # noqa : F811
                return bar(x + 1)

            @jit
            def bar(x): # noqa : F811
                return x

        msg = "The 'nopython' keyword argument was not supplied"
        self.check_warning(w, msg, NumbaDeprecationWarning)

    @TestCase.run_test_in_subprocess
    def test_guvectorize_implicit_nopython_no_warnings(self):
        # Checks that use of @guvectorize with implicit nopython compilation
        # does not warn on compilation.
        with _catch_numba_deprecation_warnings() as w:

            @guvectorize('void(float64[::1], float64[::1])', '(n)->(n)')
            def bar(a, b):
                a += 1

        self.assertFalse(w)

    @TestCase.run_test_in_subprocess
    def test_guvectorize_forceobj_no_warnings(self):
        # Checks that use of @guvectorize with direct objmode compilation does
        # not warn.
        with _catch_numba_deprecation_warnings() as w:

            @guvectorize('void(float64[::1], float64[::1])', '(n)->(n)',
                         forceobj=True)
            def bar(a, b):
                object()
                a += 1

        self.assertFalse(w)

    @TestCase.run_test_in_subprocess
    def test_guvectorize_parallel_implicit_nopython_no_warnings(self):
        # Checks that use of @guvectorize with parallel target and implicit
        # nopython mode compilation does not warn.
        with _catch_numba_deprecation_warnings() as w:

            @guvectorize('void(float64[::1], float64[::1])', '(n)->(n)',
                         target='parallel')
            def bar(a, b):
                a += 1

        self.assertFalse(w)

    @TestCase.run_test_in_subprocess
    def test_guvectorize_parallel_forceobj_no_warnings(self):
        # Checks that use of @guvectorize with parallel target and direct
        # objmode compilation does not warn.
        with _catch_numba_deprecation_warnings() as w:

            # This compiles somewhat surprisingly for the parallel target using
            # object mode?!
            @guvectorize('void(float64[::1], float64[::1])', '(n)->(n)',
                         target='parallel', forceobj=True)
            def bar(a, b):
                object()
                a += 1

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

        with _catch_numba_deprecation_warnings() as w:

            @generated_jit
            def bar():
                return lambda : None

            @jit(nopython=True)
            def foo():
                bar()

            foo()

            self.check_warning(w, "numba.generated_jit is deprecated",
                               NumbaDeprecationWarning)

    @needs_setuptools
    @TestCase.run_test_in_subprocess
    def test_pycc_module(self):
        # checks import of module warns

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always",
                                  category=NumbaPendingDeprecationWarning)
            import numba.pycc # noqa: F401

            expected_str = ("The 'pycc' module is pending deprecation.")
            self.check_warning(w, expected_str, NumbaPendingDeprecationWarning)

    @needs_setuptools
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
