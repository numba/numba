from __future__ import print_function
import itertools
import numba.unittest_support as unittest
from numba.compiler import compile_isolated
from numba import types


def identity(x):
    return x


def addition(x, y):
    return x + y


def equality(x, y):
    return x == y


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

    def test_boolean_as_int(self):
        pyfunc = equality
        cres = compile_isolated(pyfunc, [types.boolean, types.intp])
        cfunc = cres.entry_point

        xs = True, False
        ys = -1, 0, 1

        for xs, ys in itertools.product(xs, ys):
            self.assertEqual(pyfunc(xs, ys), cfunc(xs, ys))

    def test_boolean_as_float(self):
        pyfunc = equality
        cres = compile_isolated(pyfunc, [types.boolean, types.float64])
        cfunc = cres.entry_point

        xs = True, False
        ys = -1, 0, 1

        for xs, ys in itertools.product(xs, ys):
            self.assertEqual(pyfunc(xs, ys), cfunc(xs, ys))

    def test_boolean_eq_boolean(self):
        pyfunc = equality
        cres = compile_isolated(pyfunc, [types.boolean, types.boolean])
        cfunc = cres.entry_point

        xs = True, False
        ys = True, False

        for xs, ys in itertools.product(xs, ys):
            self.assertEqual(pyfunc(xs, ys), cfunc(xs, ys))

if __name__ == '__main__':
    unittest.main()
