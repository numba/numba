"""
Test problems in nested calls.
Usually due to invalid type conversion between function boundaries.
"""


from numba import cuda
from numba.core import types
from numba.cuda.testing import CUDATestCase
from numba.extending import overload
import unittest
import numpy as np


def generated_inner(out, x, y=5, z=6):
    # Provide implementation for the simulation.
    if isinstance(x, complex):
        out[0], out[1] = x + y, z
    else:
        out[0], out[1] = x - y, z


@overload(generated_inner)
def ol_generated_inner(out, x, y=5, z=6):
    if isinstance(x, types.Complex):
        def impl(out, x, y=5, z=6):
            out[0], out[1] = x + y, z
    else:
        def impl(out, x, y=5, z=6):
            out[0], out[1] = x - y, z
    return impl


def call_generated(a, b, out):
    generated_inner(out, a, z=b)


class TestNestedCall(CUDATestCase):
    def test_call_generated(self):
        """
        Test a nested function call to a generated jit function.
        """
        cfunc = cuda.jit(call_generated)

        out = np.empty(2, dtype=np.int64)
        cfunc[1,1](1, 2, out)
        self.assertPreciseEqual(tuple(out), (-4, 2))

        out = np.empty(2, dtype=np.complex64)
        cfunc[1,1](1j, 2, out)
        self.assertPreciseEqual(tuple(map(complex,out)), (5 + 1j, 2 + 0j))


if __name__ == '__main__':
    unittest.main()
