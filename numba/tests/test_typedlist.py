from __future__ import print_function, absolute_import, division

import sys

import numpy as np

from numba import njit, utils
from numba import int32, int64, float32, float64, types
from numba import typeof
from numba.typed import List
from numba.utils import IS_PY3
from numba.errors import TypingError
from .support import TestCase, MemoryLeakMixin, unittest

skip_py2 = unittest.skipUnless(IS_PY3, reason='not supported in py2')


class TestTypedList(MemoryLeakMixin, TestCase):
    def test_basic(self):
        l = List.empty_list(int32)
        # len
        self.assertEqual(len(l), 0)
        # append
        l.append(0)
        # len
        self.assertEqual(len(l), 1)
        # setitem
        l.append(0)
        l.append(0)
        l[0] = 10
        l[1] = 11
        l[2] = 12
        # getitem
        self.assertEqual(l[0], 10)
        self.assertEqual(l[1], 11)
        self.assertEqual(l[2], 12)

    def test_compiled(self):
        @njit
        def producer():
            l = List.empty_list(int32)
            l.append(23)
            return l

        @njit
        def consumer(l):
            return l[0]

        l = producer()
        val = consumer(l)
        self.assertEqual(val, 23)
