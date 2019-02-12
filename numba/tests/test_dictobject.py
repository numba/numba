"""
Testing C implementation of the numba dictionary
"""
from __future__ import print_function, absolute_import, division


from numba import njit
from numba import int32, float32
from numba import dictobject
from .support import TestCase, MemoryLeakMixin


class TestDictObject(MemoryLeakMixin, TestCase):
    def test_dict_create(self):
        @njit
        def foo():
            d = dictobject.new_dict(int32, float32)
            return len(d)

        self.assertEqual(foo(), 0)
