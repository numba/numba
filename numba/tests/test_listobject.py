"""
Testing numba implementation of the numba list.
The tests here only check that the numba typing and codegen are working
correctly.  Detailed testing of the underlying dictionary operations is done
in test_listimpl.py.
"""
from __future__ import print_function, absolute_import, division

import sys

import numpy as np

from numba import njit, utils
from numba import int32, int64, float32, float64, types
from numba import listobject, typeof
from numba.typed import Dict
from numba.utils import IS_PY3
from numba.errors import TypingError
from .support import TestCase, MemoryLeakMixin, unittest

skip_py2 = unittest.skipUnless(IS_PY3, reason='not supported in py2')


class TestListObject(MemoryLeakMixin, TestCase):
    def test_list_create(self):
        """
        Exercise list creation, append and len
        """
        @njit
        def foo(n):
            l = listobject.new_list(int32)
            for i in range(n):
                l.append(i)
            return len(l)

        self.assertEqual(foo(0), 0)
        self.assertEqual(foo(1), 1)
        self.assertEqual(foo(2), 2)
        self.assertEqual(foo(100), 100)

    def test_list_getitem(self):
        """
        Exercise list getitem
        """
        @njit
        def foo(n):
            l = listobject.new_list(int32)
            l.append(n)
            return l[0]

        self.assertEqual(foo(0), 0)
        self.assertEqual(foo(1), 1)
        self.assertEqual(foo(2), 2)
        self.assertEqual(foo(100), 100)

        # check no leak so far
        self.assert_no_memory_leak()
        # disable leak check for exception test
        self.disable_leak_check()

        @njit
        def bar():
            l = listobject.new_list(int32)
            return l[0]

        with self.assertRaises(IndexError):
            bar()
