from __future__ import absolute_import, division, print_function

from numba import unittest_support as unittest
from numba.runtime import rtsys


class Dummy(object):
    alive = 0

    def __init__(self):
        type(self).alive += 1

    def __del__(self):
        type(self).alive -= 1


class TestNrtMemInfo(unittest.TestCase):
    """
    Unitest for core MemInfo functionality
    """

    def test_meminfo_refct_1(self):
        d = Dummy()
        self.assertEqual(Dummy.alive, 1)
        addr = 0xdeadcafe  # some made up location

        mi = rtsys.meminfo_new(addr, d)
        del d
        self.assertEqual(Dummy.alive, 1)
        mi.acquire()
        self.assertEqual(Dummy.alive, 1)
        mi.release()
        del mi
        self.assertEqual(Dummy.alive, 0)

    def test_meminfo_refct_2(self):
        d = Dummy()
        self.assertEqual(Dummy.alive, 1)
        addr = 0xdeadcafe  # some made up location

        mi = rtsys.meminfo_new(addr, d)
        del d
        self.assertEqual(Dummy.alive, 1)
        for _ in range(100):
            mi.acquire()
        self.assertEqual(Dummy.alive, 1)
        for _ in range(100):
            mi.release()
        del mi
        self.assertEqual(Dummy.alive, 0)

    def test_defer_dtor(self):
        d = Dummy()
        self.assertEqual(Dummy.alive, 1)
        addr = 0xdeadcafe  # some made up location

        mi = rtsys.meminfo_new(addr, d)
        # Set defer flag
        mi.defer = True
        del d
        self.assertEqual(Dummy.alive, 1)
        mi.acquire()
        self.assertEqual(Dummy.alive, 1)
        mi.release()
        del mi
        # mi refct is zero but not yet removed due to deferring
        self.assertEqual(Dummy.alive, 1)
        rtsys.process_defer_dtor()
        self.assertEqual(Dummy.alive, 0)


if __name__ == '__main__':
    unittest.main()


