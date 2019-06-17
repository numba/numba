"""
Hash implementations for Numba types
"""
from __future__ import print_function, absolute_import, division

import math
import numpy as np
import sys

import llvmlite.llvmpy.core as lc
from llvmlite import ir

from numba.extending import (
    overload, overload_method, intrinsic, register_jitable)
from numba import types
from numba.unsafe.bytes import grab_byte, grab_uint64_t

_py34_or_later = sys.version_info[:2] >= (3, 4)

if _py34_or_later:
    # This is Py_hash_t, which is a Py_ssize_t, which has sizeof(size_t):
    # https://github.com/python/cpython/blob/d1dd6be613381b996b9071443ef081de8e5f3aff/Include/pyport.h#L91-L96
    _hash_width = sys.hash_info.width
    _Py_hash_t = getattr(types, 'int%s' % _hash_width)
    _Py_uhash_t = getattr(types, 'uint%s' % _hash_width)

    # Constants from CPython source, obtained by various means:
    # https://github.com/python/cpython/blob/d1dd6be613381b996b9071443ef081de8e5f3aff/Include/pyhash.h
    _PyHASH_INF = sys.hash_info.inf
    _PyHASH_NAN = sys.hash_info.nan
    _PyHASH_MODULUS = _Py_uhash_t(sys.hash_info.modulus)
    _PyHASH_BITS = 31 if types.intp.bitwidth == 32 else 61  # mersenne primes
    _PyHASH_MULTIPLIER = 0xf4243  # 1000003UL
    _PyHASH_IMAG = _PyHASH_MULTIPLIER
    _PyLong_SHIFT = sys.int_info.bits_per_digit
    _Py_HASH_CUTOFF = sys.hash_info.cutoff
    _Py_hashfunc_name = sys.hash_info.algorithm
else:
    _hash_width = types.intp.bitwidth
    _Py_hash_t = getattr(types, 'int%s' % _hash_width)
    _Py_uhash_t = getattr(types, 'uint%s' % _hash_width)

    # these are largely just copied in from python 3 as reasonable defaults
    _PyHASH_INF = 314159
    _PyHASH_NAN = 0
    _PyHASH_BITS = 31 if types.intp.bitwidth == 32 else 61  # mersenne primes
    _PyHASH_MODULUS = _Py_uhash_t((1 << _PyHASH_BITS) - 1)
    _PyHASH_MULTIPLIER = 0xf4243  # 1000003UL
    _PyHASH_IMAG = _PyHASH_MULTIPLIER
    _PyLong_SHIFT = 30 if types.intp.bitwidth == 64 else 15
    _Py_HASH_CUTOFF = 0
    # set this as siphash24 for py27... TODO: implement py27 string first!
    _Py_hashfunc_name = "siphash24"

# hash(obj) is implemented by calling obj.__hash__()


@overload(hash)
def hash_overload(obj):
    def impl(obj):
        return obj.__hash__()
    return impl


@register_jitable
def process_return(val):
    asint = _Py_hash_t(val)
    if (asint == int(-1)):
        asint = int(-2)
    return asint

# This is a translation of CPython's _Py_HashDouble:
# https://github.com/python/cpython/blob/d1dd6be613381b996b9071443ef081de8e5f3aff/Python/pyhash.c#L34-L129


@register_jitable(locals={'x': _Py_uhash_t,
                          'y': _Py_uhash_t,
                          'm': types.double,
                          'e': types.intc,
                          'sign': types.intc,
                          '_PyHASH_MODULUS': _Py_uhash_t,
                          '_PyHASH_BITS': types.intc})
def _Py_HashDouble(v):
    if not np.isfinite(v):
        if (np.isinf(v)):
            if (v > 0):
                return _PyHASH_INF
            else:
                return -_PyHASH_INF
        else:
            return _PyHASH_NAN

    m, e = math.frexp(v)

    sign = 1
    if (m < 0):
        sign = -1
        m = -m

    # process 28 bits at a time;  this should work well both for binary
    #  and hexadecimal floating point.
    x = 0
    while (m):
        x = ((x << 28) & _PyHASH_MODULUS) | x >> (_PyHASH_BITS - 28)
        m *= 268435456.0  # /* 2**28 */
        e -= 28
        y = int(m)  # /* pull out integer part */
        m -= y
        x += y
        if x >= _PyHASH_MODULUS:
            x -= _PyHASH_MODULUS
    # /* adjust for the exponent;  first reduce it modulo _PyHASH_BITS */
    if e >= 0:
        e = e % _PyHASH_BITS
    else:
        e = _PyHASH_BITS - 1 - ((-1 - e) % _PyHASH_BITS)

    x = ((x << e) & _PyHASH_MODULUS) | x >> (_PyHASH_BITS - e)

    x = x * sign
    return process_return(x)


@intrinsic
def _fpext(tyctx, val):
    def impl(cgctx, builder, signature, args):
        val = args[0]
        return builder.fpext(val, lc.Type.double())
    sig = types.float64(types.float32)
    return sig, impl

# This is a translation of CPython's long_hash, but restricted to the numerical
# domain reachable by int64/uint64 (i.e. no BigInt like support):
# https://github.com/python/cpython/blob/d1dd6be613381b996b9071443ef081de8e5f3aff/Objects/longobject.c#L2934-L2989
# obdigit is a uint32_t which is typedef'd to digit
# int32_t is typedef'd to sdigit


@register_jitable(locals={'x': _Py_uhash_t,
                          'p1': _Py_uhash_t,
                          'p2': _Py_uhash_t,
                          'p3': _Py_uhash_t,
                          'p4': _Py_uhash_t,
                          '_PyHASH_MODULUS': _Py_uhash_t,
                          '_PyHASH_BITS': types.int32,
                          '_PyLong_SHIFT': types.int32,
                          'x.1': _Py_uhash_t})
def _long_impl(val):
    # This function assumes val came from a long int repr with val being a
    # uint64_t this means having to split the input into PyLong_SHIFT size
    # chunks in an unsigned hash wide type, max numba can handle is a 64bit int

    # mask to select low _PyLong_SHIFT bits
    _tmp_shift = 32 - _PyLong_SHIFT
    mask_shift = (~types.uint32(0x0)) >> _tmp_shift

    # a 64bit wide max means Numba only needs 3 x 30 bit values max,
    # or 5 x 15 bit values max on 32bit platforms
    i = (64 // _PyLong_SHIFT) + 1

    # alg as per hash_long
    x = 0
    p3 = (_PyHASH_BITS - _PyLong_SHIFT)
    for idx in range(i - 1, -1, -1):
        p1 = x << _PyLong_SHIFT
        p2 = p1 & _PyHASH_MODULUS
        p4 = x >> p3
        x = p2 | p4
        # the shift and mask splits out the `ob_digit` parts of a Long repr
        x += types.uint32((val >> idx * _PyLong_SHIFT) & mask_shift)
        if x >= _PyHASH_MODULUS:
            x -= _PyHASH_MODULUS
    return _Py_hash_t(x)


# This has no CPython equivalent, CPython uses long_hash.
@overload_method(types.Integer, '__hash__')
@overload_method(types.Boolean, '__hash__')
def int_hash(val):

    _HASH_I64_MIN = -2 if sys.maxsize <= 2 ** 32 else -4

    # this is a bit involved due to the CPython repr of ints
    def impl(val):
        # If the magnitude is under PyHASH_MODULUS, if so just return the
        # value itval as the has, couple of special cases if val == val:
        # 1. it's 0, in which case return 0
        # 2. it's signed int minimum value, return the value CPython computes
        # but Numba cannot as there's no type wide enough to hold the shifts.
        #
        # If the magnitude is greater than PyHASH_MODULUS then... if the value
        # is negative then negate it switch the sign on the hash once computed
        # and use the standard wide unsigned hash implementation
        mag = abs(val)
        if mag < _PyHASH_MODULUS:
            if val == -val:
                if val == 0:
                    ret = 0
                else:  # int64 min, -0x8000000000000000
                    ret = _Py_hash_t(_HASH_I64_MIN)
            else:
                ret = _Py_hash_t(val)
        else:
            needs_negate = False
            if val < 0:
                val = -val
                needs_negate = True
            ret = _long_impl(val)
            if needs_negate:
                ret = -ret
        return process_return(ret)
    return impl

# This is a translation of CPython's float_hash:
# https://github.com/python/cpython/blob/d1dd6be613381b996b9071443ef081de8e5f3aff/Objects/floatobject.c#L528-L532


@overload_method(types.Float, '__hash__')
def float_hash(val):
    if val.bitwidth == 64:
        def impl(val):
            hashed = _Py_HashDouble(val)
            return hashed
    else:
        def impl(val):
            # widen the 32bit float to 64bit
            fpextended = np.float64(_fpext(val))
            hashed = _Py_HashDouble(fpextended)
            return hashed
    return impl

# This is a translation of CPython's complex_hash:
# https://github.com/python/cpython/blob/d1dd6be613381b996b9071443ef081de8e5f3aff/Objects/complexobject.c#L408-L428


@overload_method(types.Complex, '__hash__')
def complex_hash(val):
    def impl(val):
        hashreal = hash(val.real)
        hashimag = hash(val.imag)
        # Note:  if the imaginary part is 0, hashimag is 0 now,
        # so the following returns hashreal unchanged.  This is
        # important because numbers of different types that
        # compare equal must have the same hash value, so that
        # hash(x + 0*j) must equal hash(x).
        combined = hashreal + _PyHASH_IMAG * hashimag
        return process_return(combined)
    return impl


# This is a translation of CPython's tuplehash:
# https://github.com/python/cpython/blob/d1dd6be613381b996b9071443ef081de8e5f3aff/Objects/tupleobject.c#L347-L369
@register_jitable(locals={'x': _Py_uhash_t,
                          'y': _Py_hash_t,
                          'mult': _Py_uhash_t,
                          'l': _Py_hash_t, })
def _tuple_hash(tup):
    tl = len(tup)
    mult = _PyHASH_MULTIPLIER
    x = _Py_uhash_t(0x345678)
    # in C this is while(--l >= 0), i is indexing tup instead of *tup++
    for i, l in enumerate(range(tl - 1, -1, -1)):
        y = hash(tup[i])
        xxory = (x ^ y)
        x = xxory * mult
        mult += _Py_hash_t((_Py_uhash_t(82520) + l + l))
    x += _Py_uhash_t(97531)
    return process_return(x)

# This is an obfuscated translation of CPython's tuplehash:
# https://github.com/python/cpython/blob/d1dd6be613381b996b9071443ef081de8e5f3aff/Objects/tupleobject.c#L347-L369
# The obfuscation occurs for a heterogeneous tuple as each tuple member needs
# a potentially different hash() function calling for it. This cannot be done at
# runtime as there's no way to iterate a heterogeneous tuple, so this is
# achieved by essentially unrolling the loop over the members and inserting a
# per-type hash function call for each member, and then simply computing the
# hash value in an inlined/rolling fashion.


@intrinsic
def _tuple_hash_resolve(tyctx, val):
    def impl(cgctx, builder, signature, args):
        typingctx = cgctx.typing_context
        fnty = typingctx.resolve_value_type(hash)
        tupty, = signature.args
        tup, = args
        lty = cgctx.get_value_type(signature.return_type)
        x = ir.Constant(lty, 0x345678)
        mult = ir.Constant(lty, _PyHASH_MULTIPLIER)
        shift = ir.Constant(lty, 82520)
        tl = len(tupty)
        for i, packed in enumerate(zip(tupty.types, range(tl - 1, -1, -1))):
            ty, l = packed
            sig = fnty.get_call_type(tyctx, (ty,), {})
            impl = cgctx.get_function(fnty, sig)
            tuple_val = builder.extract_value(tup, i)
            y = impl(builder, (tuple_val,))
            xxory = builder.xor(x, y)
            x = builder.mul(xxory, mult)
            lconst = ir.Constant(lty, l)
            mult = builder.add(mult, shift)
            mult = builder.add(mult, lconst)
            mult = builder.add(mult, lconst)
        x = builder.add(x, ir.Constant(lty, 97531))
        return x
    sig = _Py_hash_t(val)
    return sig, impl


@overload_method(types.BaseTuple, '__hash__')
def tuple_hash(val):
    if isinstance(val, types.Sequence):
        def impl(val):
            return _tuple_hash(val)
        return impl
    else:
        def impl(val):
            hashed = _Py_hash_t(_tuple_hash_resolve(val))
            return process_return(hashed)
        return impl


# ------------------------------------------------------------------------------
# String/bytes hashing needs hashseed info, this is from:
# https://stackoverflow.com/a/41088757
# with thanks to Martijn Pieters
#
# Developer note:
# CPython makes use of an internal "hashsecret" which is essentially a struct
# containing some state that is set on CPython initialization and contains magic
# numbers used particularly in unicode/string hashing. This code binds to the
# Python runtime libraries in use by the current process and reads the
# "hashsecret" state so that it can be used by Numba. As this is done at runtime
# the behaviour and influence of the PYTHONHASHSEED environment variable is
# accommodated.

from ctypes import (  # noqa
    c_size_t,
    c_ubyte,
    c_uint64,
    pythonapi,
    Structure,
    Union,
)  # noqa


class FNV(Structure):
    _fields_ = [
        ('prefix', c_size_t),
        ('suffix', c_size_t)
    ]


class SIPHASH(Structure):
    _fields_ = [
        ('k0', c_uint64),
        ('k1', c_uint64),
    ]


class DJBX33A(Structure):
    _fields_ = [
        ('padding', c_ubyte * 16),
        ('suffix', c_size_t),
    ]


class EXPAT(Structure):
    _fields_ = [
        ('padding', c_ubyte * 16),
        ('hashsalt', c_size_t),
    ]


class _Py_HashSecret_t(Union):
    _fields_ = [
        # ensure 24 bytes
        ('uc', c_ubyte * 24),
        # two Py_hash_t for FNV
        ('fnv', FNV),
        # two uint64 for SipHash24
        ('siphash', SIPHASH),
        # a different (!) Py_hash_t for small string optimization
        ('djbx33a', DJBX33A),
        ('expat', EXPAT),
    ]


# Only a few members are needed at present
_Py_hashSecret = _Py_HashSecret_t.in_dll(pythonapi, '_Py_HashSecret')
_Py_HashSecret_djbx33a = _Py_hashSecret.djbx33a
_Py_HashSecret_djbx33a_suffix = _Py_hashSecret.djbx33a.suffix
_Py_HashSecret_siphash_k0 = _Py_hashSecret.siphash.k0
_Py_HashSecret_siphash_k1 = _Py_hashSecret.siphash.k1
# ------------------------------------------------------------------------------


if _Py_hashfunc_name == 'siphash24':
    # This is a translation of CPython's siphash24 function:
    # https://github.com/python/cpython/blob/d1dd6be613381b996b9071443ef081de8e5f3aff/Python/pyhash.c#L287-L413

    # /* *********************************************************************
    # <MIT License>
    # Copyright (c) 2013  Marek Majkowski <marek@popcount.org>

    # Permission is hereby granted, free of charge, to any person obtaining a copy
    # of this software and associated documentation files (the "Software"), to deal
    # in the Software without restriction, including without limitation the rights
    # to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    # copies of the Software, and to permit persons to whom the Software is
    # furnished to do so, subject to the following conditions:

    # The above copyright notice and this permission notice shall be included in
    # all copies or substantial portions of the Software.

    # THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    # IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    # FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    # AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    # LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    # OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
    # THE SOFTWARE.
    # </MIT License>

    # Original location:
    # https://github.com/majek/csiphash/

    # Solution inspired by code from:
    # Samuel Neves (supercop/crypto_auth/siphash24/little)
    #djb (supercop/crypto_auth/siphash24/little2)
    # Jean-Philippe Aumasson (https://131002.net/siphash/siphash24.c)

    # Modified for Python by Christian Heimes:
    # - C89 / MSVC compatibility
    # - _rotl64() on Windows
    # - letoh64() fallback
    # */

    @register_jitable(locals={'x': types.uint64,
                              'b': types.uint64, })
    def _ROTATE(x, b):
        return types.uint64(((x) << (b)) | ((x) >> (types.uint64(64) - (b))))

    @register_jitable(locals={'a': types.uint64,
                              'b': types.uint64,
                              'c': types.uint64,
                              'd': types.uint64,
                              's': types.uint64,
                              't': types.uint64, })
    def _HALF_ROUND(a, b, c, d, s, t):
        a += b
        c += d
        b = _ROTATE(b, s) ^ a
        d = _ROTATE(d, t) ^ c
        a = _ROTATE(a, 32)
        return a, b, c, d

    @register_jitable(locals={'v0': types.uint64,
                              'v1': types.uint64,
                              'v2': types.uint64,
                              'v3': types.uint64, })
    def _DOUBLE_ROUND(v0, v1, v2, v3):
        v0, v1, v2, v3 = _HALF_ROUND(v0, v1, v2, v3, 13, 16)
        v2, v1, v0, v3 = _HALF_ROUND(v2, v1, v0, v3, 17, 21)
        v0, v1, v2, v3 = _HALF_ROUND(v0, v1, v2, v3, 13, 16)
        v2, v1, v0, v3 = _HALF_ROUND(v2, v1, v0, v3, 17, 21)
        return v0, v1, v2, v3

    @register_jitable(locals={'v0': types.uint64,
                              'v1': types.uint64,
                              'v2': types.uint64,
                              'v3': types.uint64,
                              'b': types.uint64,
                              'mi': types.uint64,
                              'tmp': types.Array(types.uint64, 1, 'C'),
                              't': types.uint64,
                              'mask': types.uint64,
                              'jmp': types.uint64,
                              'ohexefef': types.uint64})
    def _siphash24(k0, k1, src, src_sz):
        b = types.uint64(src_sz) << 56
        v0 = k0 ^ types.uint64(0x736f6d6570736575)
        v1 = k1 ^ types.uint64(0x646f72616e646f6d)
        v2 = k0 ^ types.uint64(0x6c7967656e657261)
        v3 = k1 ^ types.uint64(0x7465646279746573)

        idx = 0
        while (src_sz >= 8):
            mi = grab_uint64_t(src, idx)
            idx += 1
            src_sz -= 8
            v3 ^= mi
            v0, v1, v2, v3 = _DOUBLE_ROUND(v0, v1, v2, v3)
            v0 ^= mi

        # this is the switch fallthrough:
        # https://github.com/python/cpython/blob/d1dd6be613381b996b9071443ef081de8e5f3aff/Python/pyhash.c#L390-L400
        t = types.uint64(0x0)
        boffset = idx * 8
        ohexefef = types.uint64(0xff)
        if src_sz >= 7:
            jmp = (6 * 8)
            mask = ~types.uint64(ohexefef << jmp)
            t = (t & mask) | (types.uint64(grab_byte(src, boffset + 6)) << jmp)
        if src_sz >= 6:
            jmp = (5 * 8)
            mask = ~types.uint64(ohexefef << jmp)
            t = (t & mask) | (types.uint64(grab_byte(src, boffset + 5)) << jmp)
        if src_sz >= 5:
            jmp = (4 * 8)
            mask = ~types.uint64(ohexefef << jmp)
            t = (t & mask) | (types.uint64(grab_byte(src, boffset + 4)) << jmp)
        if src_sz >= 4:
            t &= types.uint64(0xffffffff00000000)
            for i in range(4):
                jmp = i * 8
                mask = ~types.uint64(ohexefef << jmp)
                t = (t & mask) | (types.uint64(grab_byte(src, boffset + i)) << jmp)
        if src_sz >= 3:
            jmp = (2 * 8)
            mask = ~types.uint64(ohexefef << jmp)
            t = (t & mask) | (types.uint64(grab_byte(src, boffset + 2)) << jmp)
        if src_sz >= 2:
            jmp = (1 * 8)
            mask = ~types.uint64(ohexefef << jmp)
            t = (t & mask) | (types.uint64(grab_byte(src, boffset + 1)) << jmp)
        if src_sz >= 1:
            mask = ~(ohexefef)
            t = (t & mask) | (types.uint64(grab_byte(src, boffset + 0)))

        b |= t
        v3 ^= b
        v0, v1, v2, v3 = _DOUBLE_ROUND(v0, v1, v2, v3)
        v0 ^= b
        v2 ^= ohexefef
        v0, v1, v2, v3 = _DOUBLE_ROUND(v0, v1, v2, v3)
        v0, v1, v2, v3 = _DOUBLE_ROUND(v0, v1, v2, v3)
        t = (v0 ^ v1) ^ (v2 ^ v3)
        return t

elif _Py_hashfunc_name == 'fnv':
    #TODO: Should this instead warn and switch to siphash24?
    raise NotImplementedError("FNV hashing is not implemented")
else:
    msg = "Unsupported hashing algorithm in use %s" % _Py_hashfunc_name
    raise ValueError(msg)

# This is a translation of CPythons's _Py_HashBytes:
# https://github.com/python/cpython/blob/d1dd6be613381b996b9071443ef081de8e5f3aff/Python/pyhash.c#L145-L191


@register_jitable(locals={'_hash': _Py_uhash_t})
def _Py_HashBytes(val, _len):
    if (_len == 0):
        return process_return(0)

    if (_len < _Py_HASH_CUTOFF):
        # TODO: this branch needs testing, needs a CPython setup for it!
        # /* Optimize hashing of very small strings with inline DJBX33A. */
        _hash = _Py_uhash_t(5381)  # /* DJBX33A starts with 5381 */
        for idx in range(_len):
            _hash = ((_hash << 5) + _hash) + np.uint8(grab_byte(val, idx))

        _hash ^= _len
        _hash ^= _Py_HashSecret_djbx33a_suffix
    else:
        tmp = _siphash24(types.uint64(_Py_HashSecret_siphash_k0),
                         types.uint64(_Py_HashSecret_siphash_k1),
                         val, _len)
        _hash = process_return(tmp)
    return process_return(_hash)

# This is an approximate translation of CPython's unicode_hash:
# https://github.com/python/cpython/blob/d1dd6be613381b996b9071443ef081de8e5f3aff/Objects/unicodeobject.c#L11635-L11663


@overload_method(types.UnicodeType, '__hash__')
def unicode_hash(val):
    from numba.unicode import _kind_to_byte_width

    def impl(val):
        kindwidth = _kind_to_byte_width(val._kind)
        _len = len(val)
        # use the cache if possible
        current_hash = val._hash
        if current_hash != -1:
            return current_hash
        else:
            # cannot write hash value to cache in the unicode struct due to
            # pass by value on the struct making the struct member immutable
            return _Py_HashBytes(val._data, kindwidth * _len)

    return impl
