"""
Tests for enum support.
"""


import numpy as np
import unittest
from numba import jit, vectorize

from numba.tests.support import TestCase
from .enum_usecases import Color, Shape, Shake, Planet, RequestError


def compare_usecase(a, b):
    return a == b, a != b, a is b, a is not b


def getattr_usecase(a):
    # Lookup of a enum member on its class
    return a is Color.red


def getitem_usecase(a):
    """Lookup enum member by string name"""
    return a is Color['red']


def identity_usecase(a, b, c):
    return (a is Shake.mint,
            b is Shape.circle,
            c is RequestError.internal_error,
            )


def make_constant_usecase(const):
    def constant_usecase(a):
        return a is const
    return constant_usecase


def return_usecase(a, b, pred):
    return a if pred else b


def int_coerce_usecase(x):
    # Implicit coercion of intenums to ints
    if x > RequestError.internal_error:
        return x - RequestError.not_found
    else:
        return x + Shape.circle


def vectorize_usecase(x):
    return RequestError['internal_error']


def numpy_serial_usecase(x):
    out = np.empty_like(x)
    for i, x_ in enumerate(x):
        if x_ != RequestError.not_found:
            out[i] = RequestError['internal_error']
        else:
            out[i] = RequestError.dummy
    return out


class BaseEnumTest(object):

    def test_compare(self):
        pyfunc = compare_usecase
        cfunc = jit(nopython=True)(pyfunc)

        for args in self.pairs:
            self.assertPreciseEqual(pyfunc(*args), cfunc(*args))

    def test_return(self):
        """
        Passing and returning enum members.
        """
        pyfunc = return_usecase
        cfunc = jit(nopython=True)(pyfunc)

        for pair in self.pairs:
            for pred in (True, False):
                args = pair + (pred,)
                self.assertIs(pyfunc(*args), cfunc(*args))

    def check_constant_usecase(self, pyfunc):
        cfunc = jit(nopython=True)(pyfunc)

        for arg in self.values:
            self.assertPreciseEqual(pyfunc(arg), cfunc(arg))

    def test_constant(self):
        self.check_constant_usecase(getattr_usecase)
        self.check_constant_usecase(getitem_usecase)
        self.check_constant_usecase(make_constant_usecase(self.values[0]))

    def test_hash(self):
        pyfun = lambda x: hash(x)
        cfunc = jit(nopython=True)(pyfun)
        for member in self.values:
            self.assertPreciseEqual(pyfun(member), cfunc(member))


class TestEnum(BaseEnumTest, TestCase):
    """
    Tests for Enum classes and members.
    """
    values = [Color.red, Color.green]

    pairs = [
        (Color.red, Color.red),
        (Color.red, Color.green),
        (Shake.mint, Shake.vanilla),
        (Planet.VENUS, Planet.MARS),
        (Planet.EARTH, Planet.EARTH),
        ]

    def test_identity(self):
        """
        Enum with equal values should not compare identical
        """
        pyfunc = identity_usecase
        cfunc = jit(nopython=True)(pyfunc)
        args = (Color.blue, Color.green, Shape.square)
        self.assertPreciseEqual(pyfunc(*args), cfunc(*args))


class TestIntEnum(BaseEnumTest, TestCase):
    """
    Tests for IntEnum classes and members.
    """
    values = [Shape.circle, Shape.square]

    pairs = [
        (Shape.circle, Shape.circle),
        (Shape.circle, Shape.square),
        (RequestError.not_found, RequestError.not_found),
        (RequestError.internal_error, RequestError.not_found),
        ]

    def test_int_coerce(self):
        pyfunc = int_coerce_usecase
        cfunc = jit(nopython=True)(pyfunc)

        for arg in [300, 450, 550]:
            self.assertPreciseEqual(pyfunc(arg), cfunc(arg))

    def test_explicit_int_cast(self):
        pyfunc = lambda x: int(x)
        cfunc = jit(nopython=True)(pyfunc)
        for member in RequestError:
            self.assertPreciseEqual(pyfunc(member), cfunc(member))

    def test_array_of_enums(self):
        pyfunc = lambda : np.array([RequestError.dummy, RequestError.not_found])
        cfunc = jit(nopython=True)(pyfunc)
        self.assertPreciseEqual(pyfunc(), cfunc())

    def test_numpy_comparison(self):
        cases = [
            lambda x: RequestError.dummy == x,
            lambda x: RequestError.dummy != x,
            lambda x: RequestError.dummy < x,
            lambda x: RequestError.dummy <= x,
            lambda x: RequestError.dummy > x,
            lambda x: RequestError.dummy >= x,
        ]
        for pyfunc in cases:
            cfunc = jit(nopython=True)(pyfunc)
            arg = np.array([2, 404, 500, 404])
            self.assertPreciseEqual(pyfunc(arg), cfunc(arg))

    def test_numpy_arithmetic(self):
        cases = [
            lambda x: RequestError.dummy * x,
            lambda x: RequestError.dummy + x,
            lambda x: RequestError.dummy - x,
            lambda x: RequestError.dummy / x,
        ]
        for pyfunc in cases:
            cfunc = jit(nopython=True)(pyfunc)
            arg = np.array([2, 404, 500, 404])
            self.assertPreciseEqual(pyfunc(arg), cfunc(arg))

    def test_vectorize(self):
        cfunc = vectorize(nopython=True)(vectorize_usecase)
        arg = np.array([2, 404, 500, 404])
        sol = np.array([vectorize_usecase(i) for i in arg], dtype=arg.dtype)
        self.assertPreciseEqual(sol, cfunc(arg))

    def test_numpy_serial(self):
        cfunc = jit(nopython=True)(numpy_serial_usecase)
        arg = np.array([2, 404, 500, 404])
        sol = numpy_serial_usecase(arg)
        self.assertPreciseEqual(sol, cfunc(arg))

if __name__ == '__main__':
    unittest.main()
