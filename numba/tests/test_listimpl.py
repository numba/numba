"""
Testing C implementation of the numba list
"""
from __future__ import print_function, absolute_import, division

import ctypes
import struct

from .support import TestCase
from numba import _helperlib
from numba.config import IS_32BITS

ALIGN = 4 if IS_32BITS else 8

LIST_OK = 0


class List(object):
    """A wrapper around the C-API to provide a minimal list object for
    testing.
    """
    def __init__(self, tc, itemsize, allocated):
        """
        Parameters
        ----------
        tc : TestCase instance
        itemsize : int
            byte size for the items
        """
        self.tc = tc
        self.itemsize = itemsize
        self.allocated = allocated
        self.lp = self.list_new(itemsize, allocated)

    def __del__(self):
        self.tc.numba_list_free(self.lp)

    def __len__(self):
        return self.list_length()

    def __setitem__(self, i, item):
        return self.list_setitem(item)

    def __getitem__(self, i):
        return self.list_getitem(i)

    def append(self, item):
        self.list_append(item)

    def list_new(self, itemsize, allocated):
        lp = ctypes.c_void_p()
        status = self.tc.numba_list_new(
            ctypes.byref(lp), itemsize, allocated,
        )
        self.tc.assertEqual(status, LIST_OK)
        return lp

    def list_length(self):
        return self.tc.numba_list_length(self.lp)

    def list_setitem(self, i, item):
        status = self.tc.numba_list_setitem(self.lp, i, item)
        self.tc.assertEqual(status, LIST_OK)

    def list_getitem(self, i):
        item_out_buffer = ctypes.create_string_buffer(self.itemsize)
        status = self.tc.numba_list_getitem(self.lp, i, item_out_buffer)
        self.tc.assertEqual(status, LIST_OK)
        return item_out_buffer.raw

    def list_append(self, item):
        status = self.tc.numba_list_append(self.lp, item)
        self.tc.assertEqual(status, LIST_OK)


class TestListImpl(TestCase):
    def setUp(self):
        """Bind to the c_helper library and provide the ctypes wrapper.
        """
        list_t = ctypes.c_void_p

        def wrap(name, restype, argtypes=()):
            proto = ctypes.CFUNCTYPE(restype, *argtypes)
            return proto(_helperlib.c_helpers[name])

        # numba_list_new(NB_List *l, Py_ssize_t itemsize, Py_ssize_t allocated)
        self.numba_list_new = wrap(
            'list_new',
            ctypes.c_int,
            [ctypes.POINTER(list_t), ctypes.c_ssize_t, ctypes.c_ssize_t],
        )
        # numba_list_free(NB_List *l)
        self.numba_list_free = wrap(
            'list_free',
            None,
            [list_t],
        )
        # numba_list_length(NB_List *l)
        self.numba_list_length = wrap(
            'list_length',
            ctypes.c_int,
            [list_t],
        )
        # numba_list_setitem(NB_List *l, Py_ssize_t i, const char *item)
        self.numba_list_setitem = wrap(
            'list_setitem',
            ctypes.c_int,
            [list_t, ctypes.c_ssize_t, ctypes.c_char_p],
        )
        # numba_list_append(NB_List *l, const char *item)
        self.numba_list_append = wrap(
            'list_append',
            ctypes.c_int,
            [list_t, ctypes.c_char_p],
        )
        # numba_list_getitem(NB_List *l,  Py_ssize_t i, char *out)
        self.numba_list_getitem = wrap(
            'list_getitem',
            ctypes.c_int,
            [list_t, ctypes.c_ssize_t, ctypes.c_char_p],
        )

    def test_length(self):
        l = List(self, 8, 0)
        self.assertEqual(len(l), 0)

    def test_append_get_string(self):
        l = List(self, 8, 1)
        l.append(b"abcdefgh")
        self.assertEqual(len(l), 1)
        r = l[0]
        self.assertEqual(r, b"abcdefgh")

    def test_append_get_int(self):
        l = List(self, 8, 1)
        l.append(struct.pack("q", 1))
        self.assertEqual(len(l), 1)
        r = struct.unpack("q", l[0])[0]
        self.assertEqual(r, 1)

    def test_append_get_string_realloc(self):
        l = List(self, 8, 1)
        l.append(b"abcdefgh")
        self.assertEqual(len(l), 1)
        l.append(b"hijklmno")
        self.assertEqual(len(l), 2)
        r = l[1]
        self.assertEqual(r, b"hijklmno")
