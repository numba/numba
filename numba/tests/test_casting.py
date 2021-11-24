import numpy as np
from numba.core.compiler import compile_isolated
from numba.core.errors import TypingError
from numba import njit, cuda
from numba.core import types
import struct
import unittest

from numba.core.types.scalars import Integer

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

def float_to_int(x):
    return types.int32(x)

def float_to_int_generic(x, t):
    return t(x)


def int_to_float(x):
    return types.float64(x) / 2


def float_to_unsigned(x):
    return types.uint32(x)


def float_to_complex(x):
    return types.complex128(x)


def numpy_scalar_cast_error():
    np.int32(np.zeros((4,)))

class TestCasting(unittest.TestCase):
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

    def test_float16_to_int32(self):
        pyfunc = float_to_int
        cr = compile_isolated(pyfunc, [types.float16])
        cfunc = cr.entry_point

        self.assertEqual(cr.signature.return_type, types.int32)
        self.assertEqual(cfunc(12.3), pyfunc(12.3))
        self.assertEqual(cfunc(12.3), int(12.3))
        self.assertEqual(cfunc(-12.3), pyfunc(-12.3))
        self.assertEqual(cfunc(-12.3), int(-12.3))
    
    def test_float16_to_int16(self):
        pyfunc = float_to_int
        cr = compile_isolated(pyfunc, [types.float32])
        cfunc = cr.entry_point

        self.assertEqual(cr.signature.return_type, types.int16)
        self.assertEqual(cfunc(12.3), pyfunc(12.3))
        self.assertEqual(cfunc(12.3), np.int16(12.3))
        self.assertEqual(cfunc(-12.3), pyfunc(-12.3))
        self.assertEqual(cfunc(-12.3), np.int16(-12.3))
    
    def test_float_to_int(self):
        pyfunc = float_to_int
        cr = compile_isolated(pyfunc, [types.float32])
        cfunc = cr.entry_point

        self.assertEqual(cr.signature.return_type, types.int32)
        self.assertEqual(cfunc(12.3), pyfunc(12.3))
        self.assertEqual(cfunc(12.3), int(12.3))
        self.assertEqual(cfunc(-12.3), pyfunc(-12.3))
        self.assertEqual(cfunc(-12.3), int(-12.3))

    def test_int_to_float(self):
        pyfunc = int_to_float
        cr = compile_isolated(pyfunc, [types.int64])
        cfunc = cr.entry_point

        self.assertEqual(cr.signature.return_type, types.float64)
        self.assertEqual(cfunc(321), pyfunc(321))
        self.assertEqual(cfunc(321), 321. / 2)

    def test_float_to_unsigned(self):
        pyfunc = float_to_unsigned
        cr = compile_isolated(pyfunc, [types.float32])
        cfunc = cr.entry_point

        self.assertEqual(cr.signature.return_type, types.uint32)
        self.assertEqual(cfunc(3.21), pyfunc(3.21))
        self.assertEqual(cfunc(3.21), struct.unpack('I', struct.pack('i',
                                                                      3))[0])

    def test_float_to_complex(self):
        pyfunc = float_to_complex
        cr = compile_isolated(pyfunc, [types.float64])
        cfunc = cr.entry_point
        self.assertEqual(cr.signature.return_type, types.complex128)
        self.assertEqual(cfunc(-3.21), pyfunc(-3.21))
        self.assertEqual(cfunc(-3.21), -3.21 + 0j)

    def test_array_to_array(self):
        """Make sure this compiles.

        Cast C to A array
        """
        @njit("f8(f8[:])")
        def inner(x):
            return x[0]

        inner.disable_compile()

        @njit("f8(f8[::1])")
        def driver(x):
            return inner(x)

        x = np.array([1234], dtype=np.float64)
        self.assertEqual(driver(x), x[0])
        self.assertEqual(len(inner.overloads), 1)

    def test_0darrayT_to_T(self):
        @njit
        def inner(x):
            return x.dtype.type(x)

        inputs = [
            (np.bool_, True),
            (np.float32, 12.3),
            (np.float64, 12.3),
            (np.int64, 12),
            (np.complex64, 2j+3),
            (np.complex128, 2j+3),
            (np.timedelta64, np.timedelta64(3, 'h')),
            (np.datetime64, np.datetime64('2016-01-01')),
            ('<U3', 'ABC'),
        ]

        for (T, inp) in inputs:
            x = np.array(inp, dtype=T)
            self.assertEqual(inner(x), x[()])

    def test_array_to_scalar(self):
        """
        Ensure that a TypingError exception is raised if
        user tries to convert numpy array to scalar
        """

        with self.assertRaises(TypingError) as raises:
            compile_isolated(numpy_scalar_cast_error, ())

        self.assertIn("Casting array(float64, 1d, C) to int32 directly is unsupported.",
                      str(raises.exception))

    def test_optional_to_optional(self):
        """
        Test error due mishandling of Optional to Optional casting

        Related issue: https://github.com/numba/numba/issues/1718
        """
        # Attempt to cast optional(intp) to optional(float64)
        opt_int = types.Optional(types.intp)
        opt_flt = types.Optional(types.float64)
        sig = opt_flt(opt_int)

        @njit(sig)
        def foo(a):
            return a

        self.assertEqual(foo(2), 2)
        self.assertIsNone(foo(None))


if __name__ == '__main__':
    unittest.main()
