from __future__ import print_function
from numba import jit
import numba.unittest_support as unittest
import numpy as np


@jit
def inc(a):
    for i in xrange(len(a)):
        a[i] += 1
    return a


@jit
def inc1(a):
    a[0] += 1
    return a[0]


@jit
def inc2(a):
    a[0] += 1
    return a[0], a[0]+1


@jit
def chain1(a):
    x = y = z = inc(a)
    return x + y + z


@jit
def chain2(v):
    a = np.zeros(2)
    a[0] = x = a[1] = v
    return a[0] + a[1] + (x/2)


@jit
def unpack1(x, y):
    a, b = x, y
    return a + b/2


@jit
def unpack2(x, y):
    a, b = c, d = inc1(x), inc1(y)
    return a + c/2, b + d/2


@jit
def chain3(x, y):
    a = (b, c) = (inc1(x), inc1(y))
    (d, e) = f = (inc1(x), inc1(y))
    return (a[0] + b/2 + d + f[0]), (a[1] + c + e/2 + f[1])


@jit
def unpack3(x):
    a, b = inc2(x)
    return a + b/2


@jit
def unpack4(x):
    a, b = c, d = inc2(x)
    return a + c/2, b + d/2


@jit
def unpack5(x):
    a = b, c = inc2(x)
    d, e = f = inc2(x)
    return (a[0] + b/2 + d + f[0]), (a[1] + c + e/2 + f[1])


@jit
def unpack6(x, y):
    (a, b), (c, d) = (x, y), (y+1, x+1)
    return a + c/2, b/2 + d


class TestChainedAssign(unittest.TestCase):

    @unittest.expectedFailure
    def test_chain1(self):
        self.assertTrue(np.all(chain1(np.arange(2)) == np.array([3, 6])))
        self.assertTrue(np.all(chain1(np.arange(4, dtype=np.double)) ==
                               np.array([3.0, 6.0, 9.0, 12.0])))

    @unittest.expectedFailure
    def test_chain2(self):
        self.assertTrue(chain2(3.0) == 7.5)
        self.assertTrue(chain2(3) == 7.0)

    @unittest.expectedFailure
    def test_unpack1(self):
        self.assertTrue(unpack1(1, 3.0) == 2.5)
        self.assertTrue(unpack1(1.0, 3) == 2.0)

    @unittest.expectedFailure
    def test_unpack2(self):
        self.assertTrue(unpack2(np.array([2]), np.array([4.0])) == (4, 7.5))
        self.assertTrue(unpack2(np.array([2.0]), np.array([4])) == (4.5, 7))

    @unittest.expectedFailure
    def test_chain3(self):
        self.assertTrue(chain3(np.array([0]), np.array([1.5])) == (5, 10.25))
        self.assertTrue(chain3(np.array([0.5]), np.array([1])) == (7.25, 8))

    @unittest.expectedFailure
    def test_unpack3(self):
        self.assertTrue(unpack3(np.array([1])) == 3)
        self.assertTrue(unpack3(np.array([1.0])) == 3.5)

    @unittest.expectedFailure
    def test_unpack4(self):
        self.assertTrue(unpack4(np.array([1])) == (3, 4))
        self.assertTrue(unpack4(np.array([1.0])) == (3.0, 4.5))

    @unittest.expectedFailure
    def test_unpack5(self):
        self.assertTrue(unpack5(np.array([2])) == (12, 15))
        self.assertTrue(unpack5(np.array([2.0])) == (12.5, 15.5))

    @unittest.expectedFailure
    def test_unpack6(self):
        self.assertTrue(unpack6(3.0, 2) == (4.0, 5.0))
        self.assertTrue(unpack6(3.0, 2.0) == (4.5, 5.0))

if __name__ == '__main__':
    unittest.main()
