from __future__ import print_function, absolute_import
import numpy
from numba import config, cuda, jit
from numba.cuda.testing import unittest


def foo(ary):
    if cuda.threadIdx.x == 1:
        ary.shape[-1]


class TestException(unittest.TestCase):
    def test_exception(self):
        unsafe_foo = cuda.jit(foo)
        safe_foo = cuda.jit(debug=True)(foo)

        if not config.ENABLE_CUDASIM:
            # Simulator throws exceptions regardless of debug
            # setting
            unsafe_foo[1, 2](numpy.array([0, 1]))

        with self.assertRaises(IndexError) as cm:
            safe_foo[1, 2](numpy.array([0, 1]))
        self.assertIn("tuple index out of range", str(cm.exception))

    def test_user_raise(self):
        @cuda.jit(debug=True)
        def foo(do_raise):
            if do_raise:
                raise ValueError

        foo[1, 1](False)
        with self.assertRaises(ValueError):
            foo[1, 1](True)


if __name__ == '__main__':
    unittest.main()
