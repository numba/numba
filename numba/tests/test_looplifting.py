from __future__ import print_function, division, absolute_import
import numpy as np
from numba import unittest_support as unittest
from numba import jit


def lift1(x):
    a = np.empty(3)
    for i in range(a.size):
        a[i] = x
    return a


def lift2(x):
    a = np.empty((3, 4))
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            a[i, j] = x
    return a


def reject1(x):
    a = np.arange(4)
    for i in range(a.shape[0]):
        return a
    return a


class TestLoopLifting(unittest.TestCase):
    def test_lift1(self):
        compiled = jit(lift1)
        x = 123
        expect = lift1(x)
        got = compiled(x)
        cres = list(compiled.overloads.values())[0]
        self.assertTrue(cres.lifted)
        loopcres = list(cres.lifted[0].overloads.values())[0]
        self.assertIs(loopcres.typing_error, None)
        self.assertTrue(np.all(expect == got))

    def test_lift2(self):
        compiled = jit(lift2)
        x = 123
        expect = lift2(x)
        got = compiled(x)
        cres = list(compiled.overloads.values())[0]
        self.assertTrue(cres.lifted)
        loopcres = list(cres.lifted[0].overloads.values())[0]
        self.assertIs(loopcres.typing_error, None)
        self.assertTrue(np.all(expect == got))

    def test_reject1(self):
        compiled = jit(reject1)
        expect = reject1(1)
        got = compiled(1)
        self.assertTrue(np.all(expect == got))
        cres = list(compiled.overloads.values())[0]
        # Does not lift
        self.assertFalse(cres.lifted)


if __name__ == '__main__':
    unittest.main()
