import struct
import numpy as np

from numba import cuda
from numba.core import types
from numba.cuda.testing import CUDATestCase
import unittest


def float_to_int(x):
    return np.int32(x)


def int_to_float(x):
    return np.float64(x) / 2


def float_to_unsigned(x):
    return types.uint32(x)


def float_to_complex(x):
    return np.complex128(x)


class TestCasting(CUDATestCase):
    def _create_wrapped(self, pyfunc, intype, outtype):
        wrapped_func = cuda.jit(device=True)(pyfunc)

        @cuda.jit
        def cuda_wrapper_fn(arg, res):
            res[0] = wrapped_func(arg[0])

        def wrapper_fn(arg):
            argarray = np.zeros(1, dtype=intype)
            argarray[0] = arg
            resarray = np.zeros(1, dtype=outtype)
            cuda_wrapper_fn[1, 1](argarray, resarray)
            return resarray[0]

        return wrapper_fn

    def test_float_to_int(self):
        pyfunc = float_to_int
        cfunc = self._create_wrapped(pyfunc, np.float32, np.int32)

        self.assertEqual(cfunc(12.3), pyfunc(12.3))
        self.assertEqual(cfunc(12.3), int(12.3))
        self.assertEqual(cfunc(-12.3), pyfunc(-12.3))
        self.assertEqual(cfunc(-12.3), int(-12.3))

    def test_int_to_float(self):
        pyfunc = int_to_float
        cfunc = self._create_wrapped(pyfunc, np.int64, np.float64)

        self.assertEqual(cfunc(321), pyfunc(321))
        self.assertEqual(cfunc(321), 321. / 2)

    def test_float_to_unsigned(self):
        pyfunc = float_to_unsigned
        cfunc = self._create_wrapped(pyfunc, np.float32, np.uint32)

        self.assertEqual(cfunc(3.21), pyfunc(3.21))
        self.assertEqual(cfunc(3.21),
                         struct.unpack('I', struct.pack('i', 3))[0])

    def test_float_to_complex(self):
        pyfunc = float_to_complex
        cfunc = self._create_wrapped(pyfunc, np.float64, np.complex128)

        self.assertEqual(cfunc(-3.21), pyfunc(-3.21))
        self.assertEqual(cfunc(-3.21), -3.21 + 0j)


if __name__ == '__main__':
    unittest.main()
