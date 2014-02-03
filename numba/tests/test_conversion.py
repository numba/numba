from __future__ import print_function
import numba.unittest_support as unittest
from numba.compiler import compile_isolated, Flags
from numba import types


def identity(x):
    return x


def addition(x, y):
    return x + y


class TestConversion(unittest.TestCase):
    """
    Testing Python to Native conversion
    """
    def test_complex_identity(self):
        pyfunc = identity
        cres = compile_isolated(pyfunc, [types.complex64],
                                return_type=types.complex64)

        xs = [1.0j, (1+1j), (-1-1j), (1+0j)]
        for x in xs:
            self.assertEqual(cres.entry_point(x=x), x)


        cres = compile_isolated(pyfunc, [types.complex128],
                                return_type=types.complex128)

        xs = [1.0j, (1+1j), (-1-1j), (1+0j)]
        for x in xs:
            self.assertEqual(cres.entry_point(x=x), x)

    def test_complex_addition(self):
        pyfunc = addition
        cres = compile_isolated(pyfunc, [types.complex64, types.complex64],
                                return_type=types.complex64)

        xs = [1.0j, (1+1j), (-1-1j), (1+0j)]
        for x in xs:
            y = x
            self.assertEqual(cres.entry_point(x, y), x + y)


        cres = compile_isolated(pyfunc, [types.complex128, types.complex128],
                                return_type=types.complex128)

        xs = [1.0j, (1+1j), (-1-1j), (1+0j)]
        for x in xs:
            y = x
            self.assertEqual(cres.entry_point(x, y), x + y)


if __name__ == '__main__':
    unittest.main()
