from numba import unittest_support as unittest
import numpy as np
from numba.compiler import compile_isolated
from numba import types, njit
import struct


def float_to_int(x):
    return types.int32(x)


def int_to_float(x):
    return types.float64(x) / 2


def float_to_unsigned(x):
    return types.uint32(x)


def float_to_complex(x):
    return types.complex128(x)


class TestCasting(unittest.TestCase):
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
