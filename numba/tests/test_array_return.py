import numpy as np

from numba.core.compiler import compile_isolated
from numba import typeof
from numba.tests.support import MemoryLeakMixin
import unittest


def array_return(a, i):
    a[i] = 123
    return a


def array_return_start_with_loop(a):
    for i in range(a.size):
        a[i] += 1
    return a


class TestArrayReturn(MemoryLeakMixin, unittest.TestCase):
    def test_array_return(self):
        a = np.arange(10)
        i = 2
        at, it = typeof(a), typeof(i)
        cres = compile_isolated(array_return, (at, it))
        cfunc = cres.entry_point
        self.assertIs(a, cfunc(a, i))

    def test_array_return_start_with_loop(self):
        """
        A bug breaks array return if the function starts with a loop
        """
        a = np.arange(10)
        at = typeof(a)
        cres = compile_isolated(array_return_start_with_loop, [at])
        cfunc = cres.entry_point
        self.assertIs(a, cfunc(a))


if __name__ == '__main__':
    unittest.main()
