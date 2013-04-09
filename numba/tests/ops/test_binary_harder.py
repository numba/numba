import unittest
import numpy as np
from itertools import product
from numba import jit

def add(a, b):
    return a + b

def sub(a, b):
    return a - b

def mul(a, b):
    return a * b

def div(a, b):
    return a // b

def boundary_range(s=0):
    maxint = 2**32 - 1
    blimit = 10
    for i in xrange(s, blimit):
        yield i
    i = blimit
    while i < maxint:
        i &= maxint
        yield i
        i = i << 1
    for i in xrange(maxint - blimit, maxint):
        yield i

def boundary_range_signed(s=0):
    maxint = 2**16 - 1
    blimit = 10
    for sign in (1, -1):
        for i in xrange(s, blimit):
            yield sign * i
        i = blimit
        while i < maxint:
            yield sign * i
            i = i << 1
        for i in xrange(maxint - blimit, maxint):
            yield sign * i


class TestBinaryOps(unittest.TestCase):
    def template(self, pyfn, op, s=0):
        msg = "%s %s %s -> %s (expect %s)"
        types = 'uint32', 'float64'
        for ty in types:
            signature = '%s(%s,%s)' % (ty, ty, ty)
            fn = jit(signature)(pyfn)
            for a, b in product(boundary_range(s=s), boundary_range(s=s)):
                if ty in ['float64']:
                    a, b = float(a), float(b)

                exp = pyfn(a, b)
                if ty in ['uint32']:
                    exp &= 0xffffffff
                try:
                    got = fn(a, b)
                except:
                    print('Exception raised at fn=%s a=%s b=%s, exp=%s ty=%s' %
                          (pyfn, a, b, exp, ty))
                    raise
                self.assertTrue(exp == got, msg % (a, op, b, got, exp))

    def template2(self, pyfn, op, s=0, w=True):
        msg = "%s %s %s -> %s (expect %s)"
        types = 'int32', 'float64'
        for ty in types:
            signature = '%s(%s,%s)' % (ty, ty, ty)
            fn = jit(signature)(pyfn)
            for a, b in product(boundary_range_signed(s=s),
                                boundary_range_signed(s=s)):
                if ty in ['float64']:
                    a, b = float(a), float(b)

                if w:
                    exp = pyfn(np.asarray(a, dtype=ty), np.asarray(b, dtype=ty))
                else:
                    exp = pyfn(a, b)
                try:
                    got = fn(a, b)
                except:
                    print('Exception raised at fn=%s a=%s b=%s, exp=%s ty=%s' %
                          (pyfn, a, b, exp, ty))
                    raise
                self.assertTrue(exp == got, msg % (a, op, b, got, exp))


    def test_add(self):
        self.template(add, '+')
        self.template2(add, '+')

    def test_sub(self):
        self.template(sub, '-')
        self.template2(sub, '-')

    def test_mul(self):
        self.template(mul, '*')
        self.template2(mul, '*')

    def test_div(self):
        self.template(div, '//', s=1)
        self.template2(div, '//', s=1, w=False)

if __name__ == '__main__':
    unittest.main()
