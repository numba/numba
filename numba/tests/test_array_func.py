from numba import njit
from numba.tests.support import TestCase, MemoryLeakMixin

import numpy as np


class ArrayWrapper:
    def __init__(self, wrapped):
        self.wrapped = wrapped

    def __array__(self):
        return self.wrapped


class TestArrayFunc(MemoryLeakMixin, TestCase):
    def test_array_func(self):
        @njit
        def add(x, y):
            return x + y

        N = 5
        x = ArrayWrapper(np.arange(N))
        y = ArrayWrapper(np.ones(N))
        np.testing.assert_equal(add(x, y), np.add(x, y))
