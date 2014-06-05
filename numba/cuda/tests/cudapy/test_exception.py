from __future__ import print_function, absolute_import
import numpy
from numba import jit, cuda
from numba.cuda.testing import unittest


def foo(ary):
    if cuda.threadIdx.x == 1:
        ary.shape[-1]


class TestException(unittest.TestCase):
    def test_exception(self):
        unsafe_foo = jit(target='cuda')(foo)
        safe_foo = jit(target='cuda', debug=True)(foo)

        unsafe_foo[1, 2](numpy.array([0,1]))

        try:
            safe_foo[1, 2](numpy.array([0,1]))
        except cuda.KernelRuntimeError as e:
            print("error raised as expected:\n%s" % e)
        else:
            raise AssertionError("expecting an exception")

if __name__ == '__main__':
    unittest.main()
