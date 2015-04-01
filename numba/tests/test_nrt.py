from __future__ import absolute_import, division, print_function

from numba import unittest_support as unittest


class TestNRT(unittest.TestCase):
    def test_meminfo_op(self):
        from numba.runtime import _nrt_python as _nrt

        class Dummy(object):
            alive = 0

            def __init__(self):
                type(self).alive += 1

            def __del__(self):
                type(self).alive -= 1

        d = Dummy()
        self.assertEqual(Dummy.alive, 1)
        addr = 0xdeadcafe
        mi = _nrt.meminfo_new(addr, d)
        del d
        self.assertEqual(Dummy.alive, 1)
        _nrt.meminfo_acquire(mi)
        self.assertEqual(Dummy.alive, 1)
        _nrt.meminfo_release(mi)
        self.assertEqual(Dummy.alive, 0)


if __name__ == '__main__':
    unittest.main()


