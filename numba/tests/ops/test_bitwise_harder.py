import unittest
from itertools import product
from numba import jit

@jit('uint32(uint32, uint32)')
def bitlshift(a, b):
    return (a << b) & 0xffffffff

@jit('uint32(uint32, uint32)')
def bitrshift(a, b):
    return (a >> b) & 0xffffffff

@jit('int32(int32, int32)')
def bitashift(a, b):
    return (a >> b) & 0xffffffff

@jit('uint32(uint32, uint32)')
def bitand(a, b):
    return a & b

@jit('uint32(uint32, uint32)')
def bitor(a, b):
    return a | b

@jit('uint32(uint32)')
def bitnot(a):
    return ~a

@jit('uint32(uint32, uint32)')
def bitxor(a, b):
    return a ^ b

def loop_all(begin, end, mask):
    hash = 0
    for i in xrange(begin, end):
        hash ^= i | ((hash << 1) & mask)
    return hash

def loop_all_simpler(begin, end):
    hash = 0
    for i in xrange(begin, end):
        hash ^= begin + hash
    return hash

def boundary_range():
    maxint = 2**32 - 1
    blimit = 10
    for i in xrange(0, blimit):
        yield i
    i = blimit
    while i < maxint:
        i &= 0xffffffff
        yield i
        i = i << 1
    for i in xrange(maxint - blimit, maxint):
        yield i

class TestBitwise(unittest.TestCase):
    def template(self, testfn, expfn, op):
        for a, b in product(boundary_range(), boundary_range()):
            exp = expfn(a, b)
            try:
                got = testfn(a, b)
            except:
                print('Exception raised at a=%s b=%s, exp=%s' % (a, b, exp))
                raise
            self.assertTrue(exp == got,
                            "%s %s %s -> %s (expect %s)" % (a, op, b, got, exp))

    def test_lshift(self):
        self.template(bitlshift, bitlshift.py_func, '<<')

    def test_rshift(self):
        self.template(bitrshift, bitrshift.py_func, '>>')

    def test_ashift(self):
        self.template(bitashift, bitashift.py_func, '>>')

    def test_and(self):
        self.template(bitand, bitand.py_func, '&')

    def test_or(self):
        self.template(bitor, bitor.py_func, '|')

    def test_xor(self):
        self.template(bitxor, bitxor.py_func, '^')

class TestBitwiseLoop(unittest.TestCase):
    def test_loop_all_simpler(self):
        fn = jit('uint32(uint32, uint32)')(loop_all_simpler)
        msg = "a=%s b=%s got=%s exp=%s"
        a, b = 0, 2**16 - 1
        exp = fn.py_func(a, b)
        got = fn(a, b)
        self.assertTrue(exp == got, msg % (a, b, got, exp))

    def test_loop_all(self):
        fn = jit('uint32(uint32, uint32, uint32)')(loop_all)
        msg = "a=%s b=%s got=%s exp=%s"
        a, b = 0, 2**16 - 1
        c = 0xffffffff
        exp = fn.py_func(a, b, c)
        got = fn(a, b, c)
        self.assertTrue(exp == got, msg % (a, b, got, exp))

if __name__ == '__main__':
    unittest.main()
