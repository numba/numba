from __future__ import print_function, absolute_import
from numbapro import cuda
from numbapro.testsupport import unittest
import numpy


@cuda.autojit
def foo(x):
    i = cuda.grid(1)
    if i < x.size:
        x[i] += 1


class TestForAll(unittest.TestCase):
    def test_forall(self):
        arr = numpy.arange(11)
        orig = arr.copy()
        foo.forall(arr.size)(arr)
        self.assertTrue(numpy.all(arr == orig + 1))


if __name__ == '__main__':
    unittest.main()

