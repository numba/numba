from __future__ import print_function, absolute_import
from numbapro import cuda
from numbapro.testsupport import unittest
import numpy


@cuda.autojit
def foo(x):
    i = cuda.grid(1)
    if i < x.size:
        x[i] += 1


@cuda.jit("void(float32, float32[:], float32[:])")
def bar(a, x, y):
    i = cuda.grid(1)
    if i < x.size:
        y[i] = a * x[i] + y[i]


class TestForAll(unittest.TestCase):
    def test_forall_1(self):
        arr = numpy.arange(11)
        orig = arr.copy()
        foo.forall(arr.size)(arr)
        self.assertTrue(numpy.all(arr == orig + 1))

    def test_forall_2(self):
        x = numpy.arange(13, dtype=numpy.float32)
        y = numpy.arange(13, dtype=numpy.float32)
        oldy = y.copy()
        a = 1.234
        bar.forall(y.size)(a, x, y)
        self.assertTrue(numpy.all(y == (a * x + oldy)))


if __name__ == '__main__':
    unittest.main()

