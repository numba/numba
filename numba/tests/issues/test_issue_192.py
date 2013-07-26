import unittest
from numba import jit

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
    # TestBitwiseLoop('test_loop_all_simpler').debug()
    unittest.main()
