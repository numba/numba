"""Implements operations on bytes and str (unicode) array items."""
import operator
import numpy as np
from llvmlite import ir

from numba import njit, types, cgutils, unicode
from numba.extending import overload, intrinsic

# bytes and str arrays items are of type CharSeq and UnicodeCharSeq,
# respectively.  See numpy/types/npytypes.py for CharSeq,
# UnicodeCharSeq definitions.  The corresponding data models are
# defined in numpy/datamodel/models.py. Boxing/unboxing of item types
# are defined in numpy/targets/boxing.py, see box_unicodecharseq,
# unbox_unicodecharseq, box_charseq, unbox_charseq.

s1_dtype = np.dtype('S1')
assert s1_dtype.itemsize == 1

u1_dtype = np.dtype('U1')
unicode_byte_width = u1_dtype.itemsize
unicode_uint = {1: np.uint8, 2: np.uint16, 4: np.uint32}[unicode_byte_width]


# this is modified version of numba.unicode.make_deref_codegen
def make_deref_codegen(bitsize):
    def codegen(context, builder, signature, args):
        data, idx = args
        # XXX how to access data (of type CharSeq/UnicodeCharSeq)
        # without alloca?
        rawptr = cgutils.alloca_once_value(builder, value=data)
        ptr = builder.bitcast(rawptr, ir.IntType(bitsize).as_pointer())
        ch = builder.load(builder.gep(ptr, [idx]))
        return builder.zext(ch, ir.IntType(32))
    return codegen


@intrinsic
def deref_uint8(typingctx, data, offset):
    sig = types.uint32(data, types.intp)
    return sig, make_deref_codegen(8)


@intrinsic
def deref_uint16(typingctx, data, offset):
    sig = types.uint32(data, types.intp)
    return sig, make_deref_codegen(16)


@intrinsic
def deref_uint32(typingctx, data, offset):
    sig = types.uint32(data, types.intp)
    return sig, make_deref_codegen(32)


@njit(_nrt=False)
def charseq_get_code(a, i):
    """Access i-th item of CharSeq instance via code value
    """
    return deref_uint8(a, i)


@njit
def charseq_get_value(a, i):
    """Access i-th item of CharSeq instance via code value.

    null code is interpreted as IndexError
    """
    code = charseq_get_code(a, i)
    if code == 0:
        raise IndexError('index out of range')
    return code


@njit(_nrt=False)
def unicode_charseq_get_code(a, i):
    """Access i-th item of UnicodeCharSeq instance via code value
    """
    if unicode_byte_width == 4:
        return deref_uint32(a, i)
    elif unicode_byte_width == 2:
        return deref_uint16(a, i)
    elif unicode_byte_width == 1:
        return deref_uint8(a, i)
    else:
        raise NotImplementedError(
            'unicode_charseq_get_code: unicode_byte_width not in [1, 2, 4]')


@njit
def unicode_charseq_get_value(a, i):
    """Access i-th item of UnicodeCharSeq instance via unicode value

    null code is interpreted as IndexError
    """
    code = unicode_charseq_get_code(a, i)
    if code == 0:
        raise IndexError('index out of range')
    # Return numpy equivalent of `chr(code)`
    return np.array(code, unicode_uint).view(u1_dtype)[()]


@njit
def unicode_get_code(a, i):
    return unicode._get_code_point(a, i)


@njit
def bytes_get_code(a, i):
    return a[i]


def _get_code_impl(a):
    if isinstance(a, types.CharSeq):
        return charseq_get_code
    elif isinstance(a, types.Bytes):
        return bytes_get_code
    elif isinstance(a, types.UnicodeCharSeq):
        return unicode_charseq_get_code
    elif isinstance(a, types.UnicodeType):
        return unicode_get_code

#
#   Operators on bytes/unicode array items
#


@overload(operator.getitem)
def charseq_getitem(s, i):
    get_value = None
    if isinstance(i, types.Integer):
        if isinstance(s, types.CharSeq):
            get_value = charseq_get_value
        if isinstance(s, types.UnicodeCharSeq):
            get_value = unicode_charseq_get_value
    if get_value is not None:
        max_i = s.count
        msg = 'index out of range [0, %s]' % (max_i - 1)

        def getitem_impl(s, i):
            if i < max_i and i >= 0:
                return get_value(s, i)
            raise IndexError(msg)
        return getitem_impl


@overload(len)
def charseq_len(s):
    if isinstance(s, (types.CharSeq, types.UnicodeCharSeq)):
        get_code = _get_code_impl(s)
        n = s.count
        if n == 0:
            def len_impl(s):
                return 0

        else:
            def len_impl(s):
                # return the index of the last non-null value (numpy
                # behavior)
                i = n
                code = 0
                while code == 0:
                    i = i - 1
                    code = get_code(s, i)
                return i + 1

        return len_impl


@overload(operator.eq)
def charseq_eq(a, b):
    left_code = _get_code_impl(a)
    right_code = _get_code_impl(b)
    if left_code is not None and right_code is not None:
        def eq_impl(a, b):
            n = len(a)
            if n != len(b):
                return False
            for i in range(n):
                if left_code(a, i) != right_code(b, i):
                    return False
            return True
        return eq_impl


@overload(operator.ne)
def charseq_ne(a, b):
    left_code = _get_code_impl(a)
    right_code = _get_code_impl(b)
    if left_code is not None and right_code is not None:
        def ne_impl(a, b):
            return not (a == b)
        return ne_impl
