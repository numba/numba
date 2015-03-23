"""
Tests for numba.utils.
"""

from __future__ import print_function, absolute_import

import threading
import time

from numba import utils
from numba import unittest_support as unittest


class C(object):
    def __init__(self, value):
        self.value = value

    def __eq__(self, o):
        return self.value == o.value

    def __ne__(self, o):
        return self.value != o.value

    def __gt__(self, o):
        return self.value > o.value

class D(C):
    pass


class TestTotalOrdering(unittest.TestCase):

    def test_is_inherited(self):
        f = utils._is_inherited_from_object
        for cls in (C, D):
            self.assertFalse(f(cls, '__eq__'))
            self.assertFalse(f(cls, '__gt__'))
            self.assertFalse(f(cls, '__ne__'))
            self.assertTrue(f(cls, '__ge__'))
            self.assertTrue(f(cls, '__le__'))
            self.assertTrue(f(cls, '__lt__'))

    def check_total_ordering(self, cls):
        # Duplicate the class-under-test, to avoid mutating the original
        cls = type(cls.__name__, cls.__bases__, dict(cls.__dict__))
        cls = utils.total_ordering(cls)

        a, b, c, d = cls(10), cls(5), cls(15), cls(10)
        self.assertFalse(a < b)
        self.assertTrue(a < c)
        self.assertFalse(a < d)
        self.assertTrue(b < c)
        self.assertTrue(b < d)
        self.assertFalse(c < d)

        self.assertFalse(a <= b)
        self.assertTrue(a <= c)
        self.assertTrue(a <= d)
        self.assertTrue(b <= c)
        self.assertTrue(b <= d)
        self.assertFalse(c <= d)

        self.assertTrue(a > b)
        self.assertFalse(a > c)
        self.assertFalse(a > d)
        self.assertFalse(b > c)
        self.assertFalse(b > d)
        self.assertTrue(c > d)

        self.assertTrue(a >= b)
        self.assertFalse(a >= c)
        self.assertTrue(a >= d)
        self.assertFalse(b >= c)
        self.assertFalse(b >= d)
        self.assertTrue(c >= d)

    def test_total_ordering(self):
        self.check_total_ordering(C)

    def test_total_ordering_derived(self):
        self.check_total_ordering(D)


class TestNonReentrantLock(unittest.TestCase):

    def _lock(self):
        return utils.NonReentrantLock()

    def test_acquire_release(self):
        lock = self._lock()
        self.assertFalse(lock.is_owned())
        lock.acquire()
        self.assertTrue(lock.is_owned())
        lock.release()
        self.assertFalse(lock.is_owned())
        with self.assertRaises(RuntimeError):
            lock.release()
        lock.acquire()
        self.assertTrue(lock.is_owned())
        with self.assertRaises(RuntimeError):
            lock.acquire()
        self.assertTrue(lock.is_owned())
        lock.release()
        lock.acquire()
        lock.release()
        self.assertFalse(lock.is_owned())

    def test_multithreaded(self):
        lock = self._lock()
        errors = []

        def do_things():
            for i in range(5):
                self.assertFalse(lock.is_owned())
                lock.acquire()
                time.sleep(1e-4)
                self.assertTrue(lock.is_owned())
                lock.release()
        def wrapper():
            try:
                do_things()
            except BaseException as e:
                errors.append(e)

        threads = [threading.Thread(target=wrapper) for i in range(40)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertFalse(errors)
        self.assertFalse(lock.is_owned())

    def test_with(self):
        lock = self._lock()
        with lock:
            self.assertTrue(lock.is_owned())
            self.assertRaises(RuntimeError, lock.acquire)
        self.assertFalse(lock.is_owned())
        self.assertRaises(RuntimeError, lock.release)


if __name__ == '__main__':
    unittest.main()
