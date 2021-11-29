import struct
import numpy as np

from numba import cuda
from numba.core import types
from numba.cuda.testing import CUDATestCase
import unittest


@cuda.jit
def float16_to_float(r, h1, h2):
    i = cuda.grid(1)
    if i >= len(r):
        return
    r[i] = cuda.fp16.hadd(h1[i],  h2[i])


@cuda.jit
def float_add(r, h1, h2):
    i = cuda.grid(1)
    if i >= len(r):
        return
    r[i] = h1[i] + h2[i]


def float_to_complex(x):
    return types.complex128(x)


def float_to_int64(x):
    return np.int64(x)


def float_to_int(x):
    return np.int32(x)


def float_to_int16(x):
    return np.int16(x)


def float_to_int8(x):
    return np.int8(x)


def int_to_float(x):
    return np.float64(x) / 2


def float_to_unsigned(x):
    return types.uint32(x)


def float_to_unsigned64(x):
    return types.uint64(x)


def float_to_unsigned16(x):
    return np.uint16(x)


def float_to_unsigned8(x):
    return np.uint8(x)


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

    def test_float16_to_unsigned(self):
        pyfunc = float_to_unsigned
        cfunc = self._create_wrapped(pyfunc, np.float16, np.uint32)

        self.assertEqual(cfunc(3.21), pyfunc(3.21))
        self.assertEqual(cfunc(3.21),
                         struct.unpack('I', struct.pack('i', 3))[0])

    def test_float16_to_unsigned16(self):
        pyfunc = float_to_unsigned16
        cfunc = self._create_wrapped(pyfunc, np.float16, np.uint16)

        self.assertEqual(cfunc(3.21), pyfunc(3.21))
        self.assertEqual(cfunc(3.21),
                         struct.unpack('H', struct.pack('h', 3))[0])

    def test_float16_to_unsigned8(self):
        pyfunc = float_to_unsigned8
        cfunc = self._create_wrapped(pyfunc, np.float16, np.uint16)

        self.assertEqual(cfunc(3.21), pyfunc(3.21))
        self.assertEqual(cfunc(3.21),
                         struct.unpack('B', struct.pack('b', 3))[0])

    def test_float16_to_unsigned64(self):
        pyfunc = float_to_unsigned64
        cfunc = self._create_wrapped(pyfunc, np.float16, np.uint64)

        self.assertEqual(cfunc(3.21), pyfunc(3.21))
        self.assertEqual(cfunc(3.21),
                         struct.unpack('L', struct.pack('l', 3))[0])

    def test_float16_to_float32(self):
        np.random.seed(1)
        x = np.random.rand(10).astype(np.float16)
        y = np.random.rand(10).astype(np.float16)
        r = np.zeros(x.shape, dtype=np.float32)
        float16_to_float[1, 32](r, x, y)
        ref = (x + y).astype(np.float32)
        np.testing.assert_array_equal(r, ref)

    def test_float16_to_float64(self):
        np.random.seed(1)
        x = np.random.rand(10).astype(np.float16)
        y = np.random.rand(10).astype(np.float16)
        r = np.zeros(x.shape, dtype=np.float64)
        float16_to_float[1, 32](r, x, y)
        ref = (x + y).astype(np.float64)
        np.testing.assert_array_equal(r, ref)

    def test_float16_to_complex(self):
        pyfunc = float_to_complex
        cfunc = self._create_wrapped(pyfunc, np.float16, np.complex128)

        np.testing.assert_allclose(cfunc(-3.21), pyfunc(-3.21), rtol=0.05)
        np.testing.assert_allclose(cfunc(-3.21), -3.21 + 0j, rtol=0.05)

    def test_float32_to_float16(self):
        np.random.seed(1)
        x = np.random.rand(10).astype(np.float32)
        y = np.random.rand(10).astype(np.float32)
        r = np.zeros(x.shape, dtype=np.float16)
        float_add[1, 32](r, x, y)
        ref = (x + y).astype(np.float16)
        np.testing.assert_array_equal(r, ref)

    def test_float64_to_float16(self):
        np.random.seed(1)
        x = np.random.rand(10).astype(np.float64)
        y = np.random.rand(10).astype(np.float64)
        r = np.zeros(x.shape, dtype=np.float16)
        float_add[1, 32](r, x, y)
        ref = (x + y).astype(np.float16)
        np.testing.assert_array_equal(r, ref)

    def test_float16_to_int64(self):
        pyfunc = float_to_int64
        cfunc = self._create_wrapped(pyfunc, np.float16, np.int64)

        self.assertEqual(cfunc(12.3), pyfunc(12.3))
        self.assertEqual(cfunc(12.3), int(12.3))
        self.assertEqual(cfunc(-12.3), pyfunc(-12.3))
        self.assertEqual(cfunc(-12.3), int(-12.3))

    def test_float16_to_int32(self):
        pyfunc = float_to_int
        cfunc = self._create_wrapped(pyfunc, np.float16, np.int32)

        self.assertEqual(cfunc(12.3), pyfunc(12.3))
        self.assertEqual(cfunc(12.3), int(12.3))
        self.assertEqual(cfunc(-12.3), pyfunc(-12.3))
        self.assertEqual(cfunc(-12.3), int(-12.3))

    def test_float16_to_int16(self):
        pyfunc = float_to_int16
        cfunc = self._create_wrapped(pyfunc, np.float16, np.int16)

        self.assertEqual(cfunc(12.3), pyfunc(12.3))
        self.assertEqual(cfunc(12.3), np.int16(12.3))
        self.assertEqual(cfunc(-12.3), pyfunc(-12.3))
        self.assertEqual(cfunc(-12.3), np.int16(-12.3))

    def test_float16_to_int8(self):
        pyfunc = float_to_int8
        cfunc = self._create_wrapped(pyfunc, np.float16, np.int8)

        self.assertEqual(cfunc(12.3), pyfunc(12.3))
        self.assertEqual(cfunc(12.3), np.int16(12.3))
        self.assertEqual(cfunc(-12.3), pyfunc(-12.3))
        self.assertEqual(cfunc(-12.3), np.int16(-12.3))


if __name__ == '__main__':
    unittest.main()
