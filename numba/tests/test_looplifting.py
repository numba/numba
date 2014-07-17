from __future__ import print_function, division, absolute_import
import numpy as np

from numba import types
from numba import unittest_support as unittest
from numba.compiler import compile_isolated, Flags
from .support import TestCase


looplift_flags = Flags()
looplift_flags.set("enable_pyobject")
looplift_flags.set("enable_looplift")

pyobject_looplift_flags = looplift_flags.copy()
pyobject_looplift_flags.set("enable_pyobject_looplift")


def lift1(x):
    # Outer needs object mode because of np.empty()
    a = np.empty(3)
    for i in range(a.size):
        # Inner is nopython-compliant
        a[i] = x
    return a


def lift2(x):
    # Outer needs object mode because of np.empty()
    a = np.empty((3, 4))
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            # Inner is nopython-compliant
            a[i, j] = x
    return a


def reject1(x):
    a = np.arange(4)
    for i in range(a.shape[0]):
        # Inner returns a variable from outer "scope" => cannot loop-lift
        return a
    return a


def reject_npm1(x):
    a = np.empty(3, dtype=np.int32)
    for i in range(a.size):
        # Inner uses np.arange() => cannot loop-lift unless
        # enable_pyobject_looplift is enabled.
        a[i] = np.arange(i + 1)[i]

    return a


class TestLoopLifting(TestCase):

    def check_lift_ok(self, pyfunc, argtypes, args):
        """
        Check that pyfunc can loop-lift even in nopython mode.
        """
        cres = compile_isolated(pyfunc, argtypes,
                                flags=looplift_flags)
        # One lifted loop
        self.assertEqual(len(cres.lifted), 1)
        expected = pyfunc(*args)
        got = cres.entry_point(*args)
        self.assertTrue(np.all(expected == got))

    def check_no_lift(self, pyfunc, argtypes, args):
        """
        Check that pyfunc can't loop-lift.
        """
        cres = compile_isolated(pyfunc, argtypes,
                                flags=looplift_flags)
        self.assertFalse(cres.lifted)
        expected = pyfunc(*args)
        got = cres.entry_point(*args)
        self.assertTrue(np.all(expected == got))

    def check_no_lift_nopython(self, pyfunc, argtypes, args):
        """
        Check that pyfunc will fail loop-lifting if pyobject mode
        is disabled inside the loop, succeed otherwise.
        """
        cres = compile_isolated(pyfunc, argtypes,
                                flags=looplift_flags)
        self.assertTrue(cres.lifted)
        with self.assertTypingError():
            cres.entry_point(*args)
        cres = compile_isolated(pyfunc, argtypes,
                                flags=pyobject_looplift_flags)
        self.assertTrue(cres.lifted)
        expected = pyfunc(*args)
        got = cres.entry_point(*args)
        self.assertTrue(np.all(expected == got))

    def test_lift1(self):
        self.check_lift_ok(lift1, (types.intp,), (123,))

    def test_lift2(self):
        self.check_lift_ok(lift2, (types.intp,), (123,))

    def test_reject1(self):
        self.check_no_lift(reject1, (types.intp,), (123,))

    def test_reject_npm1(self):
        self.check_no_lift_nopython(reject_npm1, (types.intp,), (123,))


if __name__ == '__main__':
    unittest.main()
