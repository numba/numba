"""
Testing C implementation of the numba dictionary
"""
from __future__ import print_function, absolute_import, division

import ctypes
import random

from .support import TestCase
from numba import _helperlib


DKIX_EMPTY = -1
DKIX_DUMMY = -2
DKIX_ERROR = -3


class Dict(object):
    def __init__(self, tc, keysize, valsize):
        self.tc = tc
        self.keysize = keysize
        self.valsize = valsize
        self.dp = self.dict_new_minsize(keysize, valsize)

    def __len__(self):
        return self.dict_length()

    def __setitem__(self, k, v):
        bk = bytes(k.encode())
        bv = bytes(v.encode())
        self.tc.assertEqual(len(bk), self.keysize)
        self.tc.assertEqual(len(bv), self.valsize)
        self.dict_insert(bk, bv)

    def __getitem__(self, k):
        bk = bytes(k.encode())
        self.tc.assertEqual(len(bk), self.keysize)
        ix, old = self.dict_lookup(bk)
        if ix == DKIX_EMPTY:
            raise KeyError
        else:
            return old.decode()

    def __delitem__(self, k):
        bk = bytes(k.encode())
        self.tc.assertEqual(len(bk), self.keysize)
        if not self.dict_delitem(bk):
            raise KeyError(k)

    def get(self, k):
        try:
            return self[k]
        except KeyError:
            return

    def dict_new_minsize(self, key_size, val_size):
        dp = ctypes.c_void_p()
        status = self.tc.numba_dict_new_minsize(
            ctypes.byref(dp), key_size, val_size,
        )
        self.tc.assertEqual(status, 0)
        return dp

    def dict_length(self, ):
        return self.tc.numba_dict_length(self.dp)

    def dict_insert(self, key_bytes, val_bytes):
        hashval = hash(key_bytes)
        status = self.tc.numba_dict_insert_ez(
            self.dp, key_bytes, hashval, val_bytes,
        )
        self.tc.assertEqual(status, 0)

    def dict_lookup(self, key_bytes):
        hashval = hash(key_bytes)
        oldval_bytes = ctypes.create_string_buffer(self.valsize)
        ix = self.tc.numba_dict_lookup(
            self.dp, key_bytes, hashval, oldval_bytes,
        )
        self.tc.assertGreaterEqual(ix, DKIX_EMPTY)
        return ix, oldval_bytes.value

    def dict_delitem(self, key_bytes):
        ix, oldval = self.dict_lookup(key_bytes)
        if ix == DKIX_EMPTY:
            return False
        hashval = hash(key_bytes)
        status = self.tc.numba_dict_delitem_ez(self.dp, hashval, ix)
        self.tc.assertEqual(status, 0)
        return True


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
        # numba_dict_lookup(
        #       NB_Dict *d,
        #       const char *key_bytes,
        #       Py_hash_t hash,
        #       char *oldval_bytes
        # )
        cls.numba_dict_lookup = lib.numba_dict_lookup
        cls.numba_dict_lookup.argtypes = [
            dict_t,             # d
            ctypes.c_char_p,    # key_bytes
            hash_t,             # hash
            ctypes.c_char_p,    # oldval_bytes
        ]
        # numba_dict_delitem_ez(
        #     NB_Dict *d,
        #     Py_hash_t hash,
        #     Py_ssize_t ix
        # )
        cls.numba_dict_delitem_ez = lib.numba_dict_delitem_ez
        cls.numba_dict_delitem_ez.argtypes = [
            dict_t,             # d
            hash_t,             # hash
            ctypes.c_ssize_t,   # ix
        ]
        cls.numba_dict_delitem_ez.restype = ctypes.c_int

    def test_simple_c_test(self):
        self.lib._numba_test_dict()

    def test_insertion_small(self):
        d = Dict(self, 4, 8)
        self.assertEqual(len(d), 0)
        self.assertIsNone(d.get('abcd'))

        # First key
        d['abcd'] = 'beefcafe'
        self.assertEqual(len(d), 1)
        self.assertIsNotNone(d.get('abcd'))
        self.assertEqual(d['abcd'], 'beefcafe')

        # Duplicated key replaces
        d['abcd'] = 'cafe0000'
        self.assertEqual(len(d), 1)
        self.assertEqual(d['abcd'], 'cafe0000')

        # Second key
        d['abce'] = 'cafe0001'
        self.assertEqual(len(d), 2)
        self.assertEqual(d['abcd'], 'cafe0000')
        self.assertEqual(d['abce'], 'cafe0001')

        # Third key
        d['abcf'] = 'cafe0002'
        self.assertEqual(len(d), 3)
        self.assertEqual(d['abcd'], 'cafe0000')
        self.assertEqual(d['abce'], 'cafe0001')
        self.assertEqual(d['abcf'], 'cafe0002')

    def check_insertion_many(self, nmax):
        d = Dict(self, 8, 8)

        def make_key(v):
            return "key_{:04}".format(v)

        def make_val(v):
            return "val_{:04}".format(v)

        for i in range(nmax):
            d[make_key(i)] = make_val(i)
            self.assertEqual(len(d), i + 1)

        for i in range(nmax):
            self.assertEqual(d[make_key(i)], make_val(i))

    def test_insertion_many(self):
        # Around minsize
        self.check_insertion_many(nmax=7)
        self.check_insertion_many(nmax=8)
        self.check_insertion_many(nmax=9)
        # Around nmax = 32
        self.check_insertion_many(nmax=31)
        self.check_insertion_many(nmax=32)
        self.check_insertion_many(nmax=33)
        # Around nmax = 1024
        self.check_insertion_many(nmax=1023)
        self.check_insertion_many(nmax=1024)
        self.check_insertion_many(nmax=1025)
        # Around nmax = 4096
        self.check_insertion_many(nmax=4095)
        self.check_insertion_many(nmax=4096)
        self.check_insertion_many(nmax=4097)

    def test_deletion_small(self):
        d = Dict(self, 4, 8)
        self.assertEqual(len(d), 0)
        self.assertIsNone(d.get('abcd'))

        d['abcd'] = 'cafe0000'
        d['abce'] = 'cafe0001'
        d['abcf'] = 'cafe0002'

        self.assertEqual(len(d), 3)
        self.assertEqual(d['abcd'], 'cafe0000')
        self.assertEqual(d['abce'], 'cafe0001')
        self.assertEqual(d['abcf'], 'cafe0002')
        self.assertEqual(len(d), 3)

        # Delete first item
        del d['abcd']
        self.assertIsNone(d.get('abcd'))
        self.assertEqual(d['abce'], 'cafe0001')
        self.assertEqual(d['abcf'], 'cafe0002')
        self.assertEqual(len(d), 2)

        # Delete first item again
        with self.assertRaises(KeyError):
            del d['abcd']

        # Delete third
        del d['abcf']
        self.assertIsNone(d.get('abcd'))
        self.assertEqual(d['abce'], 'cafe0001')
        self.assertIsNone(d.get('abcf'))
        self.assertEqual(len(d), 1)

        # Delete second
        del d['abce']
        self.assertIsNone(d.get('abcd'))
        self.assertIsNone(d.get('abce'))
        self.assertIsNone(d.get('abcf'))
        self.assertEqual(len(d), 0)

    def check_delete_randomly(self, nmax, ndrop, nrefill, seed=0):
        random.seed(seed)

        d = Dict(self, 8, 8)
        keys = {}

        def make_key(v):
            return "key_{:04}".format(v)

        def make_val(v):
            return "val_{:04}".format(v)

        for i in range(nmax):
            d[make_key(i)] = make_val(i)

        # Fill to nmax
        for i in range(nmax):
            k = make_key(i)
            v = make_val(i)
            keys[k] = v
            self.assertEqual(d[k], v)

        self.assertEqual(len(d), nmax)

        # Randomly drop
        droplist = random.sample(list(keys), ndrop)
        remain = keys.copy()
        for i, k in enumerate(droplist, start=1):
            del d[k]
            del remain[k]
            self.assertEqual(len(d), nmax - i)
        self.assertEqual(len(d), nmax - ndrop)

        # Make sure everything dropped is gone
        for k in droplist:
            self.assertIsNone(d.get(k))

        # Make sure everything else is still here
        for k in remain:
            self.assertEqual(d[k], remain[k])

        # Refill
        for i in range(nrefill):
            k = make_key(nmax + i)
            v = make_val(nmax + i)
            remain[k] = v
            d[k] = v
            print(i)
            for k in remain:
                self.assertEqual(d[k], remain[k])

        self.assertEqual(len(remain), len(d))


        # Make sure everything else is still here
        for k in remain:
            self.assertEqual(d[k], remain[k])

    def test_delete_randomly(self):
        self.check_delete_randomly(nmax=13, ndrop=10, nrefill=31)


