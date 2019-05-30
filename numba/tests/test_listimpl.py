"""
Testing C implementation of the numba list
"""
from __future__ import print_function, absolute_import, division

import ctypes
import random

from .support import TestCase
from numba import _helperlib
from numba.config import IS_32BITS

ALIGN = 4 if IS_32BITS else 8


class List(object):
    """A wrapper around the C-API to provide a minimal list object for
    testing.
    """
    def __init__(self, tc, itemsize):
        """
        Parameters
        ----------
        tc : TestCase instance
        itemsize : int
            byte size for the items
        """
        self.tc = tc
        self.itemsize = itemsize
        self.lo = None

    def __len__(self):
        return self.list_length()

    def list_length(self):
        return self.tc.numba_list_length(self.lo)


class TestListImpl(TestCase):
    def setUp(self):
        """Bind to the c_helper library and provide the ctypes wrapper.
        """
        list_t = ctypes.c_void_p

        def wrap(name, restype, argtypes=()):
            proto = ctypes.CFUNCTYPE(restype, *argtypes)
            return proto(_helperlib.c_helpers[name])

        # numba_list_length(NB_List *l)
        self.numba_dict_length = wrap(
            'list_length',
            ctypes.c_ssize_t,
            [list_t],
        )
