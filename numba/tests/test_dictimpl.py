"""
Testing C implementation of the numba dictionary
"""
from __future__ import print_function, absolute_import, division

import ctypes

from numba import unittest_support as unittest
from .support import TestCase
from numba import _helperlib


class Dict(object):
    def __init__(self, tc, keysize, valsize):
        self.tc = tc
        self.keysize = keysize
        self.valsize = valsize
        self.dp = self.dict_new_minsize(keysize, valsize)

    def __len__(self):
        return self.dict_length(self.dp)

    def __setitem__(self, k, v):
        bk = bytes(k.encode())
        bv = bytes(v.encode())
        self.tc.assertEqual(len(bk), self.keysize)
        self.tc.assertEqual(len(bv), self.valsize)
        self.dict_insert(self.dp, bk, bv)

    def dict_new_minsize(self, key_size, val_size):
        dp = ctypes.c_void_p()
        status = self.tc.numba_dict_new_minsize(
            ctypes.byref(dp), key_size, val_size,
        )
        self.tc.assertEqual(status, 0)
        return dp

    def dict_length(self, dp):
        return self.tc.numba_dict_length(dp)

    def dict_insert(self, dp, key_bytes, val_bytes):
        hashval = hash(key_bytes)
        status = self.tc.numba_dict_insert_ez(
            dp, key_bytes, hashval, val_bytes,
        )
        self.tc.assertEqual(status, 0)


class TestDictImpl(TestCase):
    @classmethod
    def setUpClass(cls):
        lib = ctypes.CDLL(_helperlib.__file__)
        dict_t = ctypes.c_void_p
        hash_t = ctypes.c_ssize_t
        cls.lib = lib
        # numba_dict_new_minsize(
        #    NB_Dict **out,
        #    Py_ssize_t key_size,
        #    Py_ssize_t val_size
        # )
        cls.numba_dict_new_minsize = lib.numba_dict_new_minsize
        cls.numba_dict_new_minsize.argtypes = [
            ctypes.POINTER(dict_t),  # out
            ctypes.c_ssize_t,        # key_size
            ctypes.c_ssize_t,        # val_size
        ]
        cls.numba_dict_new_minsize.restype = ctypes.c_int
        # numba_dict_length(NB_Dict *d)
        cls.numba_dict_length = lib.numba_dict_length
        cls.numba_dict_length.argtypes = [dict_t]
        cls.numba_dict_length.restype = ctypes.c_int
        # numba_dict_insert_ez(
        #     NB_Dict    *d,
        #     const char *key_bytes,
        #     Py_hash_t   hash,
        #     const char *val_bytes,
        #     )
        cls.numba_dict_insert_ez = lib.numba_dict_insert_ez
        cls.numba_dict_insert_ez.argtypes = [
            dict_t,             # d
            ctypes.c_char_p,    # key_bytes
            hash_t,             # hash
            ctypes.c_char_p,    # val_bytes
        ]
        cls.numba_dict_insert_ez.restype = ctypes.c_int

    def test_simple_c_test(self):
        self.lib._numba_test_dict()

    def test_insertion(self):
        d = Dict(self, 4, 8)
        self.assertEqual(len(d), 0)

        # First key
        d['abcd'] = 'beefcafe'
        self.assertEqual(len(d), 1)

        # Duplicated key replaces
        d['abcd'] = 'cafe0000'
        self.assertEqual(len(d), 1)

        # Second key
        d['abce'] = 'cafe0001'
        self.assertEqual(len(d), 2)

        # Third key
        d['abcf'] = 'cafe0002'
        self.assertEqual(len(d), 3)

