from __future__ import print_function
from numba import jit, int_
import numba.unittest_support as unittest
import numpy as np

try:
    xrange
except NameError:
    xrange = range


@jit
def obj_loop1(A, i):
    r = 0
    for x in xrange(10):
        for y in xrange(10):
            items = A[x, y]
            r += 1
            if items == None:
                continue
            for item in items:
                print(item)
    return r


@jit
def obj_loop2(x):
    i = 0
    for elem in x:
        i += 1
        if elem > 9:
            break
    return i


@jit
def fill(a):
    for i in range(len(a)):
        a[i] += 1
    return a


@jit(int_(int_[:]))
def call_loop(a):
    s = 0
    for x in fill(a):
        s += x
    return s


class TestLoops(unittest.TestCase):

    def test_obj_loop1(self):
        self.assertTrue(obj_loop1(np.array([[None]*10]*10), 1) == 100)

    def test_obj_loop2(self):
        self.assertTrue(obj_loop2([1, 2, 3, 10]) == 4)
        self.assertTrue(obj_loop2(range(100)) == 11)

    def test_call_loop(self):
        self.assertTrue(call_loop(np.zeros(10, dtype=np.int)) == 10)


if __name__ == '__main__':
    unittest.main()
