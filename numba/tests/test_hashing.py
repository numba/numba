# -*- coding: utf-8 -*-
"""
Test hashing of various supported types.
"""

import unittest

import sys
import subprocess
from collections import defaultdict
from textwrap import dedent

import numpy as np

from numba import jit
from numba.core import types, utils
import unittest
from numba.tests.support import TestCase, tag, CompilationCache

from numba.cpython.unicode import compile_time_get_string_data
from numba.cpython import hashing


def hash_usecase(x):
    return hash(x)


class TestHashingSetup(TestCase):

    def test_warn_on_fnv(self):
        # FNV hash alg variant is not supported, check Numba warns
        work = """
        import sys
        import warnings
        from collections import namedtuple

        # hash_info is a StructSequence, mock as a named tuple
        fields = ["width", "modulus", "inf", "nan", "imag", "algorithm",
                  "hash_bits", "seed_bits", "cutoff"]

        hinfo = sys.hash_info
        FAKE_HASHINFO = namedtuple('FAKE_HASHINFO', fields)

        fd = dict()
        for f in fields:
            fd[f] = getattr(hinfo, f)

        fd['algorithm'] = 'fnv'

        fake_hashinfo = FAKE_HASHINFO(**fd)

        # replace the hashinfo with the fnv version
        sys.hash_info = fake_hashinfo
        with warnings.catch_warnings(record=True) as warns:
            # Cause all warnings to always be triggered.
            warnings.simplefilter("always")
            import numba
            assert len(warns) > 0
            expect = "FNV hashing is not implemented in Numba. See PEP 456"
            for w in warns:
                if expect in str(w.message):
                    break
            else:
                raise RuntimeError("Expected warning not found")
        """
        subprocess.check_call([sys.executable, '-c', dedent(work)])


class BaseTest(TestCase):

    def setUp(self):
        self.cfunc = jit(nopython=True)(hash_usecase)

    def check_hash_values(self, values):
        cfunc = self.cfunc
        for val in list(values):
            nb_hash = cfunc(val)
            self.assertIsInstance(nb_hash, utils.INT_TYPES)
            try:
                self.assertEqual(nb_hash, hash(val))
            except AssertionError as e:
                print("val, nb_hash, hash(val)")
                print(val, nb_hash, hash(val))
                print("abs(val), hashing._PyHASH_MODULUS - 1")
                print(abs(val), hashing._PyHASH_MODULUS - 1)
                raise e

    def int_samples(self, typ=np.int64):
        for start in (0, -50, 60000, 1 << 32):
            info = np.iinfo(typ)
            if not info.min <= start <= info.max:
                continue
            n = 100
            yield range(start, start + n)
            yield range(start, start + 100 * n, 100)
            yield range(start, start + 128 * n, 128)
            yield [-1]

    def float_samples(self, typ):
        info = np.finfo(typ)

        for start in (0, 10, info.max ** 0.5, info.max / 1000.0):
            n = 100
            min_step = max(info.tiny, start * info.resolution)
            for step in (1.2, min_step ** 0.5, min_step):
                if step < min_step:
                    continue
                a = np.linspace(start, start + n * step, n)
                a = a.astype(typ)
                yield a
                yield -a
                yield a + a.mean()

        # Infs, nans, zeros, magic -1
        a = typ([0.0, 0.5, -0.0, -1.0, float('inf'), -float('inf'),
                 float('nan')])
        yield a

    def complex_samples(self, typ, float_ty):
        for real in self.float_samples(float_ty):
            for imag in self.float_samples(float_ty):
                # Ensure equal sizes
                real = real[:len(imag)]
                imag = imag[:len(real)]
                a = real + typ(1j) * imag
                yield a


class TestNumberHashing(BaseTest):
    """
    Test hashing of number types.
    """

    def check_floats(self, typ):
        for a in self.float_samples(typ):
            self.assertEqual(a.dtype, np.dtype(typ))
            self.check_hash_values(a)

    def check_complex(self, typ, float_ty):
        for a in self.complex_samples(typ, float_ty):
            self.assertEqual(a.dtype, np.dtype(typ))
            self.check_hash_values(a)

    def test_floats(self):
        self.check_floats(np.float32)
        self.check_floats(np.float64)

    def test_complex(self):
        self.check_complex(np.complex64, np.float32)
        self.check_complex(np.complex128, np.float64)

    def test_bool(self):
        self.check_hash_values([False, True])

    def test_ints(self):
        minmax = []
        for ty in [np.int8, np.uint8, np.int16, np.uint16,
                   np.int32, np.uint32, np.int64, np.uint64]:
            for a in self.int_samples(ty):
                self.check_hash_values(a)
            info = np.iinfo(ty)
            # check hash(-1) = -2
            # check hash(0) = 0
            self.check_hash_values([ty(-1)])
            self.check_hash_values([ty(0)])
            signed = 'uint' not in str(ty)
            # check bit shifting patterns from min through to max
            sz = ty().itemsize
            for x in [info.min, info.max]:
                shifts = 8 * sz
                # x is a python int, do shifts etc as a python int and init
                # numpy type from that to avoid numpy type rules
                y = x
                for i in range(shifts):
                    twiddle1 = 0xaaaaaaaaaaaaaaaa
                    twiddle2 = 0x5555555555555555
                    vals = [y]
                    for tw in [twiddle1, twiddle2]:
                        val = y & twiddle1
                        if val < sys.maxsize:
                            vals.append(val)
                    for v in vals:
                        self.check_hash_values([ty(v)])
                    if signed:  # try the same with flipped signs
                        # negated signed INT_MIN will overflow
                        for v in vals:
                            if v != info.min:
                                self.check_hash_values([ty(-v)])
                    if x == 0:  # unsigned min is 0, shift up
                        y = (y | 1) << 1
                    else:  # everything else shift down
                        y = y >> 1

        # these straddle the branch between returning the int as the hash and
        # doing the PyLong hash alg
        self.check_hash_values([np.int64(0x1ffffffffffffffe)])
        self.check_hash_values([np.int64(0x1fffffffffffffff)])
        self.check_hash_values([np.uint64(0x1ffffffffffffffe)])
        self.check_hash_values([np.uint64(0x1fffffffffffffff)])

        # check some values near sys int mins
        self.check_hash_values([np.int64(-0x7fffffffffffffff)])
        self.check_hash_values([np.int64(-0x7ffffffffffffff6)])
        self.check_hash_values([np.int64(-0x7fffffffffffff9c)])
        self.check_hash_values([np.int32(-0x7fffffff)])
        self.check_hash_values([np.int32(-0x7ffffff6)])
        self.check_hash_values([np.int32(-0x7fffff9c)])


class TestTupleHashing(BaseTest):
    """
    Test hashing of tuples.
    """

    def check_tuples(self, value_generator, split):
        for values in value_generator:
            tuples = [split(a) for a in values]
            self.check_hash_values(tuples)

    def test_homogeneous_tuples(self):
        typ = np.uint64

        def split2(i):
            """
            Split i's bits into 2 integers.
            """
            i = typ(i)
            return (i & typ(0x5555555555555555),
                    i & typ(0xaaaaaaaaaaaaaaaa),
                    )

        def split3(i):
            """
            Split i's bits into 3 integers.
            """
            i = typ(i)
            return (i & typ(0x2492492492492492),
                    i & typ(0x4924924924924924),
                    i & typ(0x9249249249249249),
                    )

        self.check_tuples(self.int_samples(), split2)
        self.check_tuples(self.int_samples(), split3)

        # Check exact. Sample values from:
        # https://github.com/python/cpython/blob/b738237d6792acba85b1f6e6c8993a812c7fd815/Lib/test/test_tuple.py#L80-L93
        # Untypable empty tuples are replaced with (7,).
        self.check_hash_values([(7,), (0,), (0, 0), (0.5,),
                                (0.5, (7,), (-2, 3, (4, 6)))])

    def test_heterogeneous_tuples(self):
        modulo = 2**63

        def split(i):
            a = i & 0x5555555555555555
            b = (i & 0xaaaaaaaa) ^ ((i >> 32) & 0xaaaaaaaa)
            return np.int64(a), np.float64(b * 0.0001)

        self.check_tuples(self.int_samples(), split)


class TestUnicodeHashing(BaseTest):

    def test_basic_unicode(self):
        kind1_string = "abcdefghijklmnopqrstuvwxyz"
        for i in range(len(kind1_string)):
            self.check_hash_values([kind1_string[:i]])

        sep = "眼"
        kind2_string = sep.join(list(kind1_string))
        for i in range(len(kind2_string)):
            self.check_hash_values([kind2_string[:i]])

        sep = "🐍⚡"
        kind4_string = sep.join(list(kind1_string))
        for i in range(len(kind4_string)):
            self.check_hash_values([kind4_string[:i]])

        empty_string = ""
        self.check_hash_values(empty_string)

    def test_hash_passthrough(self):
        # no `hash` call made, this just checks that `._hash` is correctly
        # passed through from an already existing string
        kind1_string = "abcdefghijklmnopqrstuvwxyz"

        @jit(nopython=True)
        def fn(x):
            return x._hash

        hash_value = compile_time_get_string_data(kind1_string)[-1]
        self.assertTrue(hash_value != -1)
        self.assertEqual(fn(kind1_string), hash_value)

    def test_hash_passthrough_call(self):
        # check `x._hash` and hash(x) are the same
        kind1_string = "abcdefghijklmnopqrstuvwxyz"

        @jit(nopython=True)
        def fn(x):
            return x._hash, hash(x)

        hash_value = compile_time_get_string_data(kind1_string)[-1]
        self.assertTrue(hash_value != -1)
        self.assertEqual(fn(kind1_string), (hash_value, hash_value))

    @unittest.skip("Needs hash computation at const unpickling time")
    def test_hash_literal(self):
        # a strconst always seem to have an associated hash value so the hash
        # member of the returned value should contain the correct hash
        @jit(nopython=True)
        def fn():
            x = "abcdefghijklmnopqrstuvwxyz"
            return x
        val = fn()
        tmp = hash("abcdefghijklmnopqrstuvwxyz")
        self.assertEqual(tmp, (compile_time_get_string_data(val)[-1]))

    def test_hash_on_str_creation(self):
        # In cPython some? new strings do not have a cached hash until hash() is
        # called
        def impl(do_hash):
            const1 = "aaaa"
            const2 = "眼眼眼眼"
            new = const1 + const2
            if do_hash:
                hash(new)
            return new

        jitted = jit(nopython=True)(impl)

        # do not compute the hash, cPython will have no cached hash, but Numba
        # will
        compute_hash = False
        expected = impl(compute_hash)
        got = jitted(compute_hash)
        a = (compile_time_get_string_data(expected))
        b = (compile_time_get_string_data(got))
        self.assertEqual(a[:-1], b[:-1])
        self.assertTrue(a[-1] != b[-1])

        # now with compute hash enabled, cPython will have a cached hash as will
        # Numba
        compute_hash = True
        expected = impl(compute_hash)
        got = jitted(compute_hash)
        a = (compile_time_get_string_data(expected))
        b = (compile_time_get_string_data(got))
        self.assertEqual(a, b)


if __name__ == "__main__":
    unittest.main()
