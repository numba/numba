from __future__ import print_function, absolute_import

import numpy as np

from numba import config, cuda, jit
from numba.cuda.testing import unittest


def foo(ary):
    x = cuda.threadIdx.x
    if x == 1:
        # NOTE: indexing with a out-of-bounds constant can fail at
        # compile-time instead (because the getitem is rewritten as a static_getitem)
        # XXX: -1 is actually a valid index for a non-empty tuple...
        ary.shape[-x]


class TestException(unittest.TestCase):
    def test_exception(self):
        unsafe_foo = cuda.jit(foo)
        safe_foo = cuda.jit(debug=True)(foo)

        if not config.ENABLE_CUDASIM:
            # Simulator throws exceptions regardless of debug
            # setting
            unsafe_foo[1, 2](np.array([0, 1]))

        with self.assertRaises(IndexError) as cm:
            safe_foo[1, 2](np.array([0, 1]))
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
