from numba import unittest_support as unittest
from .base import TestCase

try:
    from xnd import xnd
    from gumath import sin

    from numba.xnd import vectorize
    from numba.xnd.gufunc import GuFunc
except ImportError:
    pass


class TestVectorize(TestCase):
    def setUp(self):
        self.fn = lambda a: a

        self.t = 'int64 -> int64'
        self.t2 = 'float64 -> float64'
        self.x = xnd(1)
        self.x2 = xnd(1.0)


    def test_just_function(self):
        gufunc = vectorize(self.fn)

        self.assertEqual(type(gufunc), GuFunc)
        self.assertEqual(len(gufunc.already_compiled), 0)
        self.assertEquals(gufunc(self.x), self.x)
        self.assertEqual(len(gufunc.already_compiled), 1)
        self.assertEquals(gufunc(self.x), self.x)
        self.assertEqual(len(gufunc.already_compiled), 1)
        self.assertEquals(gufunc(self.x2), self.x2)
        self.assertEqual(len(gufunc.already_compiled), 2)

    def test_type_string(self):
        gufunc = vectorize(self.t)(self.fn)

        self.assertEqual(type(gufunc), type(sin))
        self.assertSequenceEqual(gufunc.kernels, [self.t])
        self.assertEquals(gufunc(self.x), self.x)

    def test_type_strings(self):
        gufunc = vectorize([self.t, self.t2])(self.fn)

        self.assertEqual(type(gufunc), type(sin))
        self.assertSetEqual(set(gufunc.kernels), {self.t, self.t2})
        self.assertEquals(gufunc(self.x), self.x)
        self.assertEquals(gufunc(self.x2), self.x2)

if __name__ == '__main__':
    unittest.main()
