"""
This module contains support functions for more advanced unicode operations.
This is not a public API and is for Numba internal use only. Most of the
functions are relatively straightforward translations of the functions with the
same name in CPython.
"""

from collections import namedtuple
from enum import IntEnum

import llvmlite.ir
import numpy as np

from numba.core import types, cgutils, config
from numba.core.imputils import impl_ret_untracked

from numba.core.extending import overload, intrinsic, register_jitable
from numba.core.errors import TypingError

# This is equivalent to the struct `_PyUnicode_TypeRecord defined in CPython's
# Objects/unicodectype.c
typerecord = namedtuple("typerecord", "upper lower title decimal digit flags")

# The Py_UCS4 type from CPython:
# https://github.com/python/cpython/blob/1d4b6ba19466aba0eb91c4ba01ba509acf18c723/Include/unicodeobject.h#L112    # noqa: E501
if config.USE_LEGACY_TYPE_SYSTEM:
    _Py_UCS4 = types.uint32
else:
    _Py_UCS4 = types.c_uint32

# ------------------------------------------------------------------------------
# Start code related to/from CPython's unicodectype impl
#
# NOTE: the original source at:
# https://github.com/python/cpython/blob/1d4b6ba19466aba0eb91c4ba01ba509acf18c723/Objects/unicodectype.c   # noqa: E501
# contains this statement:
#
# /*
#   Unicode character type helpers.
#
#   Written by Marc-Andre Lemburg (mal@lemburg.com).
#   Modified for Python 2.0 by Fredrik Lundh (fredrik@pythonware.com)
#
#   Copyright (c) Corporation for National Research Initiatives.
#
# */


# This enum contains the values defined in CPython's Objects/unicodectype.c that
# provide masks for use against the various members of the typerecord
#
# See: https://github.com/python/cpython/blob/1d4b6ba19466aba0eb91c4ba01ba509acf18c723/Objects/unicodectype.c#L13-L27    # noqa: E501
#


_Py_TAB = 0x9
_Py_LINEFEED = 0xA
_Py_CARRIAGE_RETURN = 0xD
_Py_SPACE = 0x20


class _PyUnicode_TyperecordMasks(IntEnum):
    ALPHA_MASK = 0x01
    DECIMAL_MASK = 0x02
    DIGIT_MASK = 0x04
    LOWER_MASK = 0x08
    LINEBREAK_MASK = 0x10
    SPACE_MASK = 0x20
    TITLE_MASK = 0x40
    UPPER_MASK = 0x80
    XID_START_MASK = 0x100
    XID_CONTINUE_MASK = 0x200
    PRINTABLE_MASK = 0x400
    NUMERIC_MASK = 0x800
    CASE_IGNORABLE_MASK = 0x1000
    CASED_MASK = 0x2000
    EXTENDED_CASE_MASK = 0x4000


def _PyUnicode_gettyperecord(a):
    raise RuntimeError("Calling the Python definition is invalid")


@intrinsic
def _gettyperecord_impl(typingctx, codepoint):
    """
    Provides the binding to numba_gettyperecord, returns a `typerecord`
    namedtuple of properties from the codepoint.
    """
    if not isinstance(codepoint, types.Integer):
        raise TypingError("codepoint must be an integer")

    def details(context, builder, signature, args):
        ll_void = context.get_value_type(types.void)
        ll_Py_UCS4 = context.get_value_type(_Py_UCS4)
        ll_intc = context.get_value_type(types.intc)
        ll_intc_ptr = ll_intc.as_pointer()
        ll_uchar = context.get_value_type(types.uchar)
        ll_uchar_ptr = ll_uchar.as_pointer()
        ll_ushort = context.get_value_type(types.ushort)
        ll_ushort_ptr = ll_ushort.as_pointer()
        fnty = llvmlite.ir.FunctionType(
            ll_void,
            [
                ll_Py_UCS4,  # code
                ll_intc_ptr,  # upper
                ll_intc_ptr,  # lower
                ll_intc_ptr,  # title
                ll_uchar_ptr,  # decimal
                ll_uchar_ptr,  # digit
                ll_ushort_ptr,  # flags
            ],
        )
        fn = cgutils.get_or_insert_function(
            builder.module, fnty, name="numba_gettyperecord"
        )
        upper = cgutils.alloca_once(builder, ll_intc, name="upper")
        lower = cgutils.alloca_once(builder, ll_intc, name="lower")
        title = cgutils.alloca_once(builder, ll_intc, name="title")
        decimal = cgutils.alloca_once(builder, ll_uchar, name="decimal")
        digit = cgutils.alloca_once(builder, ll_uchar, name="digit")
        flags = cgutils.alloca_once(builder, ll_ushort, name="flags")

        byref = [upper, lower, title, decimal, digit, flags]
        builder.call(fn, [args[0]] + byref)
        buf = []
        for x in byref:
            buf.append(builder.load(x))

        res = context.make_tuple(builder, signature.return_type, tuple(buf))
        return impl_ret_untracked(context, builder, signature.return_type, res)

    tupty = types.NamedTuple(
        [types.intc, types.intc, types.intc, types.uchar, types.uchar, types.ushort],
        typerecord,
    )
    sig = tupty(_Py_UCS4)
    return sig, details


@overload(_PyUnicode_gettyperecord)
def gettyperecord_impl(a):
    """
    Provides a _PyUnicode_gettyperecord binding, for convenience it will accept
    single character strings and code points.
    """
    if isinstance(a, types.UnicodeType):
        from numba.cpython.unicode import _get_code_point

        def impl(a):
            if len(a) > 1:
                msg = "gettyperecord takes a single unicode character"
                raise ValueError(msg)
            code_point = _get_code_point(a, 0)
            data = _gettyperecord_impl(_Py_UCS4(code_point))
            return data

        return impl
    if isinstance(a, types.Integer):
        return lambda a: _gettyperecord_impl(_Py_UCS4(a))


# whilst it's possible to grab the _PyUnicode_ExtendedCase symbol as it's global
# it is safer to use a defined api:
@intrinsic
def _PyUnicode_ExtendedCase(typingctx, index):
    """
    Accessor function for the _PyUnicode_ExtendedCase array, binds to
    numba_get_PyUnicode_ExtendedCase which wraps the array and does the lookup
    """
    if not isinstance(index, types.Integer):
        raise TypingError("Expected an index")

    def details(context, builder, signature, args):
        ll_Py_UCS4 = context.get_value_type(_Py_UCS4)
        ll_intc = context.get_value_type(types.intc)
        fnty = llvmlite.ir.FunctionType(ll_Py_UCS4, [ll_intc])
        fn = cgutils.get_or_insert_function(
            builder.module, fnty, name="numba_get_PyUnicode_ExtendedCase"
        )
        return builder.call(fn, [args[0]])

    sig = _Py_UCS4(types.intc)
    return sig, details


# The following functions are replications of the functions with the same name
# in CPython's Objects/unicodectype.c


# From: https://github.com/python/cpython/blob/1d4b6ba19466aba0eb91c4ba01ba509acf18c723/Objects/unicodectype.c#L64-L71    # noqa: E501
@register_jitable
def _PyUnicode_ToTitlecase(ch):
    ctype = _PyUnicode_gettyperecord(ch)
    if ctype.flags & _PyUnicode_TyperecordMasks.EXTENDED_CASE_MASK:
        return _PyUnicode_ExtendedCase(ctype.title & 0xFFFF)
    return ch + ctype.title


# From: https://github.com/python/cpython/blob/1d4b6ba19466aba0eb91c4ba01ba509acf18c723/Objects/unicodectype.c#L76-L81    # noqa: E501
@register_jitable
def _PyUnicode_IsTitlecase(ch):
    ctype = _PyUnicode_gettyperecord(ch)
    return ctype.flags & _PyUnicode_TyperecordMasks.TITLE_MASK != 0


# From: https://github.com/python/cpython/blob/1d4b6ba19466aba0eb91c4ba01ba509acf18c723/Objects/unicodectype.c#L86-L91    # noqa: E501
@register_jitable
def _PyUnicode_IsXidStart(ch):
    ctype = _PyUnicode_gettyperecord(ch)
    return ctype.flags & _PyUnicode_TyperecordMasks.XID_START_MASK != 0


# From: https://github.com/python/cpython/blob/1d4b6ba19466aba0eb91c4ba01ba509acf18c723/Objects/unicodectype.c#L96-L101    # noqa: E501
@register_jitable
def _PyUnicode_IsXidContinue(ch):
    ctype = _PyUnicode_gettyperecord(ch)
    return ctype.flags & _PyUnicode_TyperecordMasks.XID_CONTINUE_MASK != 0


@register_jitable
def _PyUnicode_ToDecimalDigit(ch):
    ctype = _PyUnicode_gettyperecord(ch)
    if ctype.flags & _PyUnicode_TyperecordMasks.DECIMAL_MASK:
        return ctype.decimal
    return -1


# From: https://github.com/python/cpython/blob/1d4b6ba19466aba0eb91c4ba01ba509acf18c723/Objects/unicodectype.c#L123-L1128  # noqa: E501
@register_jitable
def _PyUnicode_ToDigit(ch):
    ctype = _PyUnicode_gettyperecord(ch)
    if ctype.flags & _PyUnicode_TyperecordMasks.DIGIT_MASK:
        return ctype.digit
    return -1


# From: https://github.com/python/cpython/blob/1d4b6ba19466aba0eb91c4ba01ba509acf18c723/Objects/unicodectype.c#L140-L145    # noqa: E501
@register_jitable
def _PyUnicode_IsNumeric(ch):
    ctype = _PyUnicode_gettyperecord(ch)
    return ctype.flags & _PyUnicode_TyperecordMasks.NUMERIC_MASK != 0


# From: https://github.com/python/cpython/blob/1d4b6ba19466aba0eb91c4ba01ba509acf18c723/Objects/unicodectype.c#L160-L165    # noqa: E501
@register_jitable
def _PyUnicode_IsPrintable(ch):
    ctype = _PyUnicode_gettyperecord(ch)
    return ctype.flags & _PyUnicode_TyperecordMasks.PRINTABLE_MASK != 0


# From: https://github.com/python/cpython/blob/1d4b6ba19466aba0eb91c4ba01ba509acf18c723/Objects/unicodectype.c#L170-L175    # noqa: E501
@register_jitable
def _PyUnicode_IsLowercase(ch):
    ctype = _PyUnicode_gettyperecord(ch)
    return ctype.flags & _PyUnicode_TyperecordMasks.LOWER_MASK != 0


# From: https://github.com/python/cpython/blob/1d4b6ba19466aba0eb91c4ba01ba509acf18c723/Objects/unicodectype.c#L180-L185    # noqa: E501
@register_jitable
def _PyUnicode_IsUppercase(ch):
    ctype = _PyUnicode_gettyperecord(ch)
    return ctype.flags & _PyUnicode_TyperecordMasks.UPPER_MASK != 0


@register_jitable
def _PyUnicode_IsLineBreak(ch):
    ctype = _PyUnicode_gettyperecord(ch)
    return ctype.flags & _PyUnicode_TyperecordMasks.LINEBREAK_MASK != 0


@register_jitable
def _PyUnicode_ToUppercase(ch):
    raise NotImplementedError


@register_jitable
def _PyUnicode_ToLowercase(ch):
    raise NotImplementedError


# From: https://github.com/python/cpython/blob/201c8f79450628241574fba940e08107178dc3a5/Objects/unicodectype.c#L211-L225    # noqa: E501
@register_jitable
def _PyUnicode_ToLowerFull(ch, res):
    ctype = _PyUnicode_gettyperecord(ch)
    if ctype.flags & _PyUnicode_TyperecordMasks.EXTENDED_CASE_MASK:
        index = ctype.lower & 0xFFFF
        n = ctype.lower >> 24
        for i in range(n):
            res[i] = _PyUnicode_ExtendedCase(index + i)
        return n
    res[0] = ch + ctype.lower
    return 1


# From: https://github.com/python/cpython/blob/201c8f79450628241574fba940e08107178dc3a5/Objects/unicodectype.c#L227-L241    # noqa: E501
@register_jitable
def _PyUnicode_ToTitleFull(ch, res):
    ctype = _PyUnicode_gettyperecord(ch)
    if ctype.flags & _PyUnicode_TyperecordMasks.EXTENDED_CASE_MASK:
        index = ctype.title & 0xFFFF
        n = ctype.title >> 24
        for i in range(n):
            res[i] = _PyUnicode_ExtendedCase(index + i)
        return n
    res[0] = ch + ctype.title
    return 1


# From: https://github.com/python/cpython/blob/1d4b6ba19466aba0eb91c4ba01ba509acf18c723/Objects/unicodectype.c#L243-L257    # noqa: E501
@register_jitable
def _PyUnicode_ToUpperFull(ch, res):
    ctype = _PyUnicode_gettyperecord(ch)
    if ctype.flags & _PyUnicode_TyperecordMasks.EXTENDED_CASE_MASK:
        index = ctype.upper & 0xFFFF
        n = ctype.upper >> 24
        for i in range(n):
            # Perhaps needed to use unicode._set_code_point() here
            res[i] = _PyUnicode_ExtendedCase(index + i)
        return n
    res[0] = ch + ctype.upper
    return 1


# From: https://github.com/python/cpython/blob/1d4b6ba19466aba0eb91c4ba01ba509acf18c723/Objects/unicodectype.c#L259-L272    # noqa: E501
@register_jitable
def _PyUnicode_ToFoldedFull(ch, res):
    ctype = _PyUnicode_gettyperecord(ch)
    extended_case_mask = _PyUnicode_TyperecordMasks.EXTENDED_CASE_MASK
    if ctype.flags & extended_case_mask and (ctype.lower >> 20) & 7:
        index = (ctype.lower & 0xFFFF) + (ctype.lower >> 24)
        n = (ctype.lower >> 20) & 7
        for i in range(n):
            res[i] = _PyUnicode_ExtendedCase(index + i)
        return n
    return _PyUnicode_ToLowerFull(ch, res)


# From: https://github.com/python/cpython/blob/1d4b6ba19466aba0eb91c4ba01ba509acf18c723/Objects/unicodectype.c#L274-L279    # noqa: E501
@register_jitable
def _PyUnicode_IsCased(ch):
    ctype = _PyUnicode_gettyperecord(ch)
    return ctype.flags & _PyUnicode_TyperecordMasks.CASED_MASK != 0


# From: https://github.com/python/cpython/blob/1d4b6ba19466aba0eb91c4ba01ba509acf18c723/Objects/unicodectype.c#L281-L286    # noqa: E501
@register_jitable
def _PyUnicode_IsCaseIgnorable(ch):
    ctype = _PyUnicode_gettyperecord(ch)
    return ctype.flags & _PyUnicode_TyperecordMasks.CASE_IGNORABLE_MASK != 0


# From: https://github.com/python/cpython/blob/1d4b6ba19466aba0eb91c4ba01ba509acf18c723/Objects/unicodectype.c#L123-L135    # noqa: E501
@register_jitable
def _PyUnicode_IsDigit(ch):
    if _PyUnicode_ToDigit(ch) < 0:
        return 0
    return 1


# From: https://github.com/python/cpython/blob/1d4b6ba19466aba0eb91c4ba01ba509acf18c723/Objects/unicodectype.c#L106-L118    # noqa: E501
@register_jitable
def _PyUnicode_IsDecimalDigit(ch):
    if _PyUnicode_ToDecimalDigit(ch) < 0:
        return 0
    return 1


# From: https://github.com/python/cpython/blob/1d4b6ba19466aba0eb91c4ba01ba509acf18c723/Objects/unicodectype.c#L291-L296    # noqa: E501
@register_jitable
def _PyUnicode_IsSpace(ch):
    ctype = _PyUnicode_gettyperecord(ch)
    return ctype.flags & _PyUnicode_TyperecordMasks.SPACE_MASK != 0


@register_jitable
def _PyUnicode_IsAlpha(ch):
    ctype = _PyUnicode_gettyperecord(ch)
    return ctype.flags & _PyUnicode_TyperecordMasks.ALPHA_MASK != 0


# End code related to/from CPython's unicodectype impl
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
# Start code related to/from CPython's pyctype


# From the definition in CPython's Include/pyctype.h
# From: https://github.com/python/cpython/blob/1d4b6ba19466aba0eb91c4ba01ba509acf18c723/Include/pyctype.h#L5-L11    # noqa: E501
class _PY_CTF(IntEnum):
    LOWER = 0x01
    UPPER = 0x02
    ALPHA = 0x01 | 0x02
    DIGIT = 0x04
    ALNUM = 0x01 | 0x02 | 0x04
    SPACE = 0x08
    XDIGIT = 0x10


# From the definition in CPython's Python/pyctype.c
# https://github.com/python/cpython/blob/1d4b6ba19466aba0eb91c4ba01ba509acf18c723/Python/pyctype.c#L5    # noqa: E501
_Py_ctype_table = np.array(
    [
        0,  # 0x0 '\x00'
        0,  # 0x1 '\x01'
        0,  # 0x2 '\x02'
        0,  # 0x3 '\x03'
        0,  # 0x4 '\x04'
        0,  # 0x5 '\x05'
        0,  # 0x6 '\x06'
        0,  # 0x7 '\x07'
        0,  # 0x8 '\x08'
        _PY_CTF.SPACE,  # 0x9 '\t'
        _PY_CTF.SPACE,  # 0xa '\n'
        _PY_CTF.SPACE,  # 0xb '\v'
        _PY_CTF.SPACE,  # 0xc '\f'
        _PY_CTF.SPACE,  # 0xd '\r'
        0,  # 0xe '\x0e'
        0,  # 0xf '\x0f'
        0,  # 0x10 '\x10'
        0,  # 0x11 '\x11'
        0,  # 0x12 '\x12'
        0,  # 0x13 '\x13'
        0,  # 0x14 '\x14'
        0,  # 0x15 '\x15'
        0,  # 0x16 '\x16'
        0,  # 0x17 '\x17'
        0,  # 0x18 '\x18'
        0,  # 0x19 '\x19'
        0,  # 0x1a '\x1a'
        0,  # 0x1b '\x1b'
        0,  # 0x1c '\x1c'
        0,  # 0x1d '\x1d'
        0,  # 0x1e '\x1e'
        0,  # 0x1f '\x1f'
        _PY_CTF.SPACE,  # 0x20 ' '
        0,  # 0x21 '!'
        0,  # 0x22 '"'
        0,  # 0x23 '#'
        0,  # 0x24 '$'
        0,  # 0x25 '%'
        0,  # 0x26 '&'
        0,  # 0x27 "'"
        0,  # 0x28 '('
        0,  # 0x29 ')'
        0,  # 0x2a '*'
        0,  # 0x2b '+'
        0,  # 0x2c ','
        0,  # 0x2d '-'
        0,  # 0x2e '.'
        0,  # 0x2f '/'
        _PY_CTF.DIGIT | _PY_CTF.XDIGIT,  # 0x30 '0'
        _PY_CTF.DIGIT | _PY_CTF.XDIGIT,  # 0x31 '1'
        _PY_CTF.DIGIT | _PY_CTF.XDIGIT,  # 0x32 '2'
        _PY_CTF.DIGIT | _PY_CTF.XDIGIT,  # 0x33 '3'
        _PY_CTF.DIGIT | _PY_CTF.XDIGIT,  # 0x34 '4'
        _PY_CTF.DIGIT | _PY_CTF.XDIGIT,  # 0x35 '5'
        _PY_CTF.DIGIT | _PY_CTF.XDIGIT,  # 0x36 '6'
        _PY_CTF.DIGIT | _PY_CTF.XDIGIT,  # 0x37 '7'
        _PY_CTF.DIGIT | _PY_CTF.XDIGIT,  # 0x38 '8'
        _PY_CTF.DIGIT | _PY_CTF.XDIGIT,  # 0x39 '9'
        0,  # 0x3a ':'
        0,  # 0x3b ';'
        0,  # 0x3c '<'
        0,  # 0x3d '='
        0,  # 0x3e '>'
        0,  # 0x3f '?'
        0,  # 0x40 '@'
        _PY_CTF.UPPER | _PY_CTF.XDIGIT,  # 0x41 'A'
        _PY_CTF.UPPER | _PY_CTF.XDIGIT,  # 0x42 'B'
        _PY_CTF.UPPER | _PY_CTF.XDIGIT,  # 0x43 'C'
        _PY_CTF.UPPER | _PY_CTF.XDIGIT,  # 0x44 'D'
        _PY_CTF.UPPER | _PY_CTF.XDIGIT,  # 0x45 'E'
        _PY_CTF.UPPER | _PY_CTF.XDIGIT,  # 0x46 'F'
        _PY_CTF.UPPER,  # 0x47 'G'
        _PY_CTF.UPPER,  # 0x48 'H'
        _PY_CTF.UPPER,  # 0x49 'I'
        _PY_CTF.UPPER,  # 0x4a 'J'
        _PY_CTF.UPPER,  # 0x4b 'K'
        _PY_CTF.UPPER,  # 0x4c 'L'
        _PY_CTF.UPPER,  # 0x4d 'M'
        _PY_CTF.UPPER,  # 0x4e 'N'
        _PY_CTF.UPPER,  # 0x4f 'O'
        _PY_CTF.UPPER,  # 0x50 'P'
        _PY_CTF.UPPER,  # 0x51 'Q'
        _PY_CTF.UPPER,  # 0x52 'R'
        _PY_CTF.UPPER,  # 0x53 'S'
        _PY_CTF.UPPER,  # 0x54 'T'
        _PY_CTF.UPPER,  # 0x55 'U'
        _PY_CTF.UPPER,  # 0x56 'V'
        _PY_CTF.UPPER,  # 0x57 'W'
        _PY_CTF.UPPER,  # 0x58 'X'
        _PY_CTF.UPPER,  # 0x59 'Y'
        _PY_CTF.UPPER,  # 0x5a 'Z'
        0,  # 0x5b '['
        0,  # 0x5c '\\'
        0,  # 0x5d ']'
        0,  # 0x5e '^'
        0,  # 0x5f '_'
        0,  # 0x60 '`'
        _PY_CTF.LOWER | _PY_CTF.XDIGIT,  # 0x61 'a'
        _PY_CTF.LOWER | _PY_CTF.XDIGIT,  # 0x62 'b'
        _PY_CTF.LOWER | _PY_CTF.XDIGIT,  # 0x63 'c'
        _PY_CTF.LOWER | _PY_CTF.XDIGIT,  # 0x64 'd'
        _PY_CTF.LOWER | _PY_CTF.XDIGIT,  # 0x65 'e'
        _PY_CTF.LOWER | _PY_CTF.XDIGIT,  # 0x66 'f'
        _PY_CTF.LOWER,  # 0x67 'g'
        _PY_CTF.LOWER,  # 0x68 'h'
        _PY_CTF.LOWER,  # 0x69 'i'
        _PY_CTF.LOWER,  # 0x6a 'j'
        _PY_CTF.LOWER,  # 0x6b 'k'
        _PY_CTF.LOWER,  # 0x6c 'l'
        _PY_CTF.LOWER,  # 0x6d 'm'
        _PY_CTF.LOWER,  # 0x6e 'n'
        _PY_CTF.LOWER,  # 0x6f 'o'
        _PY_CTF.LOWER,  # 0x70 'p'
        _PY_CTF.LOWER,  # 0x71 'q'
        _PY_CTF.LOWER,  # 0x72 'r'
        _PY_CTF.LOWER,  # 0x73 's'
        _PY_CTF.LOWER,  # 0x74 't'
        _PY_CTF.LOWER,  # 0x75 'u'
        _PY_CTF.LOWER,  # 0x76 'v'
        _PY_CTF.LOWER,  # 0x77 'w'
        _PY_CTF.LOWER,  # 0x78 'x'
        _PY_CTF.LOWER,  # 0x79 'y'
        _PY_CTF.LOWER,  # 0x7a 'z'
        0,  # 0x7b '{'
        0,  # 0x7c '|'
        0,  # 0x7d '}'
        0,  # 0x7e '~'
        0,  # 0x7f '\x7f'
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    ],
    dtype=np.intc,
)


# From the definition in CPython's Python/pyctype.c
# https://github.com/python/cpython/blob/1d4b6ba19466aba0eb91c4ba01ba509acf18c723/Python/pyctype.c#L145    # noqa: E501
_Py_ctype_tolower = np.array(
    [
        0x00,
        0x01,
        0x02,
        0x03,
        0x04,
        0x05,
        0x06,
        0x07,
        0x08,
        0x09,
        0x0A,
        0x0B,
        0x0C,
        0x0D,
        0x0E,
        0x0F,
        0x10,
        0x11,
        0x12,
        0x13,
        0x14,
        0x15,
        0x16,
        0x17,
        0x18,
        0x19,
        0x1A,
        0x1B,
        0x1C,
        0x1D,
        0x1E,
        0x1F,
        0x20,
        0x21,
        0x22,
        0x23,
        0x24,
        0x25,
        0x26,
        0x27,
        0x28,
        0x29,
        0x2A,
        0x2B,
        0x2C,
        0x2D,
        0x2E,
        0x2F,
        0x30,
        0x31,
        0x32,
        0x33,
        0x34,
        0x35,
        0x36,
        0x37,
        0x38,
        0x39,
        0x3A,
        0x3B,
        0x3C,
        0x3D,
        0x3E,
        0x3F,
        0x40,
        0x61,
        0x62,
        0x63,
        0x64,
        0x65,
        0x66,
        0x67,
        0x68,
        0x69,
        0x6A,
        0x6B,
        0x6C,
        0x6D,
        0x6E,
        0x6F,
        0x70,
        0x71,
        0x72,
        0x73,
        0x74,
        0x75,
        0x76,
        0x77,
        0x78,
        0x79,
        0x7A,
        0x5B,
        0x5C,
        0x5D,
        0x5E,
        0x5F,
        0x60,
        0x61,
        0x62,
        0x63,
        0x64,
        0x65,
        0x66,
        0x67,
        0x68,
        0x69,
        0x6A,
        0x6B,
        0x6C,
        0x6D,
        0x6E,
        0x6F,
        0x70,
        0x71,
        0x72,
        0x73,
        0x74,
        0x75,
        0x76,
        0x77,
        0x78,
        0x79,
        0x7A,
        0x7B,
        0x7C,
        0x7D,
        0x7E,
        0x7F,
        0x80,
        0x81,
        0x82,
        0x83,
        0x84,
        0x85,
        0x86,
        0x87,
        0x88,
        0x89,
        0x8A,
        0x8B,
        0x8C,
        0x8D,
        0x8E,
        0x8F,
        0x90,
        0x91,
        0x92,
        0x93,
        0x94,
        0x95,
        0x96,
        0x97,
        0x98,
        0x99,
        0x9A,
        0x9B,
        0x9C,
        0x9D,
        0x9E,
        0x9F,
        0xA0,
        0xA1,
        0xA2,
        0xA3,
        0xA4,
        0xA5,
        0xA6,
        0xA7,
        0xA8,
        0xA9,
        0xAA,
        0xAB,
        0xAC,
        0xAD,
        0xAE,
        0xAF,
        0xB0,
        0xB1,
        0xB2,
        0xB3,
        0xB4,
        0xB5,
        0xB6,
        0xB7,
        0xB8,
        0xB9,
        0xBA,
        0xBB,
        0xBC,
        0xBD,
        0xBE,
        0xBF,
        0xC0,
        0xC1,
        0xC2,
        0xC3,
        0xC4,
        0xC5,
        0xC6,
        0xC7,
        0xC8,
        0xC9,
        0xCA,
        0xCB,
        0xCC,
        0xCD,
        0xCE,
        0xCF,
        0xD0,
        0xD1,
        0xD2,
        0xD3,
        0xD4,
        0xD5,
        0xD6,
        0xD7,
        0xD8,
        0xD9,
        0xDA,
        0xDB,
        0xDC,
        0xDD,
        0xDE,
        0xDF,
        0xE0,
        0xE1,
        0xE2,
        0xE3,
        0xE4,
        0xE5,
        0xE6,
        0xE7,
        0xE8,
        0xE9,
        0xEA,
        0xEB,
        0xEC,
        0xED,
        0xEE,
        0xEF,
        0xF0,
        0xF1,
        0xF2,
        0xF3,
        0xF4,
        0xF5,
        0xF6,
        0xF7,
        0xF8,
        0xF9,
        0xFA,
        0xFB,
        0xFC,
        0xFD,
        0xFE,
        0xFF,
    ],
    dtype=np.uint8,
)


# From the definition in CPython's Python/pyctype.c
# https://github.com/python/cpython/blob/1d4b6ba19466aba0eb91c4ba01ba509acf18c723/Python/pyctype.c#L180
_Py_ctype_toupper = np.array(
    [
        0x00,
        0x01,
        0x02,
        0x03,
        0x04,
        0x05,
        0x06,
        0x07,
        0x08,
        0x09,
        0x0A,
        0x0B,
        0x0C,
        0x0D,
        0x0E,
        0x0F,
        0x10,
        0x11,
        0x12,
        0x13,
        0x14,
        0x15,
        0x16,
        0x17,
        0x18,
        0x19,
        0x1A,
        0x1B,
        0x1C,
        0x1D,
        0x1E,
        0x1F,
        0x20,
        0x21,
        0x22,
        0x23,
        0x24,
        0x25,
        0x26,
        0x27,
        0x28,
        0x29,
        0x2A,
        0x2B,
        0x2C,
        0x2D,
        0x2E,
        0x2F,
        0x30,
        0x31,
        0x32,
        0x33,
        0x34,
        0x35,
        0x36,
        0x37,
        0x38,
        0x39,
        0x3A,
        0x3B,
        0x3C,
        0x3D,
        0x3E,
        0x3F,
        0x40,
        0x41,
        0x42,
        0x43,
        0x44,
        0x45,
        0x46,
        0x47,
        0x48,
        0x49,
        0x4A,
        0x4B,
        0x4C,
        0x4D,
        0x4E,
        0x4F,
        0x50,
        0x51,
        0x52,
        0x53,
        0x54,
        0x55,
        0x56,
        0x57,
        0x58,
        0x59,
        0x5A,
        0x5B,
        0x5C,
        0x5D,
        0x5E,
        0x5F,
        0x60,
        0x41,
        0x42,
        0x43,
        0x44,
        0x45,
        0x46,
        0x47,
        0x48,
        0x49,
        0x4A,
        0x4B,
        0x4C,
        0x4D,
        0x4E,
        0x4F,
        0x50,
        0x51,
        0x52,
        0x53,
        0x54,
        0x55,
        0x56,
        0x57,
        0x58,
        0x59,
        0x5A,
        0x7B,
        0x7C,
        0x7D,
        0x7E,
        0x7F,
        0x80,
        0x81,
        0x82,
        0x83,
        0x84,
        0x85,
        0x86,
        0x87,
        0x88,
        0x89,
        0x8A,
        0x8B,
        0x8C,
        0x8D,
        0x8E,
        0x8F,
        0x90,
        0x91,
        0x92,
        0x93,
        0x94,
        0x95,
        0x96,
        0x97,
        0x98,
        0x99,
        0x9A,
        0x9B,
        0x9C,
        0x9D,
        0x9E,
        0x9F,
        0xA0,
        0xA1,
        0xA2,
        0xA3,
        0xA4,
        0xA5,
        0xA6,
        0xA7,
        0xA8,
        0xA9,
        0xAA,
        0xAB,
        0xAC,
        0xAD,
        0xAE,
        0xAF,
        0xB0,
        0xB1,
        0xB2,
        0xB3,
        0xB4,
        0xB5,
        0xB6,
        0xB7,
        0xB8,
        0xB9,
        0xBA,
        0xBB,
        0xBC,
        0xBD,
        0xBE,
        0xBF,
        0xC0,
        0xC1,
        0xC2,
        0xC3,
        0xC4,
        0xC5,
        0xC6,
        0xC7,
        0xC8,
        0xC9,
        0xCA,
        0xCB,
        0xCC,
        0xCD,
        0xCE,
        0xCF,
        0xD0,
        0xD1,
        0xD2,
        0xD3,
        0xD4,
        0xD5,
        0xD6,
        0xD7,
        0xD8,
        0xD9,
        0xDA,
        0xDB,
        0xDC,
        0xDD,
        0xDE,
        0xDF,
        0xE0,
        0xE1,
        0xE2,
        0xE3,
        0xE4,
        0xE5,
        0xE6,
        0xE7,
        0xE8,
        0xE9,
        0xEA,
        0xEB,
        0xEC,
        0xED,
        0xEE,
        0xEF,
        0xF0,
        0xF1,
        0xF2,
        0xF3,
        0xF4,
        0xF5,
        0xF6,
        0xF7,
        0xF8,
        0xF9,
        0xFA,
        0xFB,
        0xFC,
        0xFD,
        0xFE,
        0xFF,
    ],
    dtype=np.uint8,
)


class _PY_CTF_LB(IntEnum):
    LINE_BREAK = 0x01
    LINE_FEED = 0x02
    CARRIAGE_RETURN = 0x04


_Py_ctype_islinebreak = np.array(
    [
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        _PY_CTF_LB.LINE_BREAK | _PY_CTF_LB.LINE_FEED,  # 0xa '\n'
        _PY_CTF_LB.LINE_BREAK,  # 0xb '\v'
        _PY_CTF_LB.LINE_BREAK,  # 0xc '\f'
        _PY_CTF_LB.LINE_BREAK | _PY_CTF_LB.CARRIAGE_RETURN,  # 0xd '\r'
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        _PY_CTF_LB.LINE_BREAK,  # 0x1c '\x1c'
        _PY_CTF_LB.LINE_BREAK,  # 0x1d '\x1d'
        _PY_CTF_LB.LINE_BREAK,  # 0x1e '\x1e'
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        _PY_CTF_LB.LINE_BREAK,  # 0x85 '\x85'
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    ],
    dtype=np.intc,
)


# Translation of:
# https://github.com/python/cpython/blob/1d4b6ba19466aba0eb91c4ba01ba509acf18c723/Include/pymacro.h#L25    # noqa: E501
@register_jitable
def _Py_CHARMASK(ch):
    """
    Equivalent to the CPython macro `Py_CHARMASK()`, masks off all but the
    lowest 256 bits of ch.
    """
    return types.uint8(ch) & types.uint8(0xFF)


# Translation of:
# https://github.com/python/cpython/blob/1d4b6ba19466aba0eb91c4ba01ba509acf18c723/Include/pyctype.h#L30    # noqa: E501
@register_jitable
def _Py_TOUPPER(ch):
    """
    Equivalent to the CPython macro `Py_TOUPPER()` converts an ASCII range
    code point to the upper equivalent
    """
    return _Py_ctype_toupper[_Py_CHARMASK(ch)]


# Translation of:
# https://github.com/python/cpython/blob/1d4b6ba19466aba0eb91c4ba01ba509acf18c723/Include/pyctype.h#L29    # noqa: E501
@register_jitable
def _Py_TOLOWER(ch):
    """
    Equivalent to the CPython macro `Py_TOLOWER()` converts an ASCII range
    code point to the lower equivalent
    """
    return _Py_ctype_tolower[_Py_CHARMASK(ch)]


# Translation of:
# https://github.com/python/cpython/blob/1d4b6ba19466aba0eb91c4ba01ba509acf18c723/Include/pyctype.h#L18    # noqa: E501
@register_jitable
def _Py_ISLOWER(ch):
    """
    Equivalent to the CPython macro `Py_ISLOWER()`
    """
    return _Py_ctype_table[_Py_CHARMASK(ch)] & _PY_CTF.LOWER


# Translation of:
# https://github.com/python/cpython/blob/1d4b6ba19466aba0eb91c4ba01ba509acf18c723/Include/pyctype.h#L19    # noqa: E501
@register_jitable
def _Py_ISUPPER(ch):
    """
    Equivalent to the CPython macro `Py_ISUPPER()`
    """
    return _Py_ctype_table[_Py_CHARMASK(ch)] & _PY_CTF.UPPER


# Translation of:
# https://github.com/python/cpython/blob/1d4b6ba19466aba0eb91c4ba01ba509acf18c723/Include/pyctype.h#L20    # noqa: E501
@register_jitable
def _Py_ISALPHA(ch):
    """
    Equivalent to the CPython macro `Py_ISALPHA()`
    """
    return _Py_ctype_table[_Py_CHARMASK(ch)] & _PY_CTF.ALPHA


# Translation of:
# https://github.com/python/cpython/blob/1d4b6ba19466aba0eb91c4ba01ba509acf18c723/Include/pyctype.h#L21    # noqa: E501
@register_jitable
def _Py_ISDIGIT(ch):
    """
    Equivalent to the CPython macro `Py_ISDIGIT()`
    """
    return _Py_ctype_table[_Py_CHARMASK(ch)] & _PY_CTF.DIGIT


# Translation of:
# https://github.com/python/cpython/blob/1d4b6ba19466aba0eb91c4ba01ba509acf18c723/Include/pyctype.h#L22    # noqa: E501
@register_jitable
def _Py_ISXDIGIT(ch):
    """
    Equivalent to the CPython macro `Py_ISXDIGIT()`
    """
    return _Py_ctype_table[_Py_CHARMASK(ch)] & _PY_CTF.XDIGIT


# Translation of:
# https://github.com/python/cpython/blob/1d4b6ba19466aba0eb91c4ba01ba509acf18c723/Include/pyctype.h#L23    # noqa: E501
@register_jitable
def _Py_ISALNUM(ch):
    """
    Equivalent to the CPython macro `Py_ISALNUM()`
    """
    return _Py_ctype_table[_Py_CHARMASK(ch)] & _PY_CTF.ALNUM


# Translation of:
# https://github.com/python/cpython/blob/1d4b6ba19466aba0eb91c4ba01ba509acf18c723/Include/pyctype.h#L24    # noqa: E501
@register_jitable
def _Py_ISSPACE(ch):
    """
    Equivalent to the CPython macro `Py_ISSPACE()`
    """
    return _Py_ctype_table[_Py_CHARMASK(ch)] & _PY_CTF.SPACE


@register_jitable
def _Py_ISLINEBREAK(ch):
    """Check if character is ASCII line break"""
    return _Py_ctype_islinebreak[_Py_CHARMASK(ch)] & _PY_CTF_LB.LINE_BREAK


@register_jitable
def _Py_ISLINEFEED(ch):
    """Check if character is line feed `\n`"""
    return _Py_ctype_islinebreak[_Py_CHARMASK(ch)] & _PY_CTF_LB.LINE_FEED


@register_jitable
def _Py_ISCARRIAGERETURN(ch):
    """Check if character is carriage return `\r`"""
    return _Py_ctype_islinebreak[_Py_CHARMASK(ch)] & _PY_CTF_LB.CARRIAGE_RETURN


# End code related to/from CPython's pyctype
# ------------------------------------------------------------------------------
