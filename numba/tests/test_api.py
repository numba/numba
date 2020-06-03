import numba
from numba import jit, njit

from numba.tests.support import TestCase
import unittest


class TestNumbaModule(TestCase):
    """
    Test the APIs exposed by the top-level `numba` module.
    """

    def check_member(self, name):
        self.assertTrue(hasattr(numba, name), name)
        self.assertIn(name, numba.__all__)

    def test_numba_module(self):
        # jit
        self.check_member("jit")
        self.check_member("vectorize")
        self.check_member("guvectorize")
        self.check_member("njit")
        # errors
        self.check_member("NumbaError")
        self.check_member("TypingError")
        # types
        self.check_member("int32")
        # misc
        numba.__version__  # not in __all__


class TestJitDecorator(TestCase):
    """
    Test the jit and njit decorators
    """
    def test_jit_nopython_forceobj(self):
        with self.assertRaises(ValueError):
            jit(nopython=True, forceobj=True)

        def py_func(x):
            return(x)

        jit_func = jit(nopython=True)(py_func)
        jit_func(1)
        self.assertTrue(jit_func.nopython_signatures)

        jit_func = jit(forceobj=True)(py_func)
        jit_func(1)
        self.assertFalse(jit_func.nopython_signatures)

    def test_njit_nopython_forceobj(self):
        with self.assertWarns(RuntimeWarning):
            njit(forceobj=True)

        with self.assertWarns(RuntimeWarning):
            njit(nopython=True)

        def py_func(x):
            return(x)

        jit_func = njit(nopython=True)(py_func)
        jit_func(1)
        self.assertTrue(jit_func.nopython_signatures)

        jit_func = njit(forceobj=True)(py_func)
        jit_func(1)
        self.assertTrue(jit_func.nopython_signatures)


if __name__ == '__main__':
    unittest.main()
