"""Implementation of numpy.char routines."""

from numba.extending import overload, register_jitable
from numba import types
import numpy as np


# -----------------------------------------------------------------------------
# Support Functions


@register_jitable
def _register_scalar_bytes(b, rstrip=True):
    """Expose the numerical representation of scalar ASCII bytes."""
    len_chr = 1
    size_chr = len(b)
    if rstrip and size_chr > 1:
        return (
            _rstrip_inner(np.frombuffer(b, 'uint8').copy(), size_chr),
            len_chr,
            size_chr
        )
    return np.frombuffer(b, 'uint8'), len_chr, size_chr


@register_jitable
def _register_array_bytes(b, rstrip=True):
    """Expose the numerical representation of ASCII array bytes."""
    len_chr = b.size
    size_chr = b.itemsize
    if rstrip and size_chr > 1:
        return (
            _rstrip_inner(np.frombuffer(b, 'uint8').copy(), size_chr),
            len_chr,
            size_chr
        )
    return np.frombuffer(b, 'uint8'), len_chr, size_chr


@register_jitable
def _register_scalar_strings(s, rstrip=True):
    """Expose the numerical representation of scalar UTF-32 strings."""
    len_chr = 1
    size_chr = len(s)
    chr_array = np.empty(size_chr, 'int32')
    for i in range(size_chr):
        chr_array[i] = ord(s[i])
    if rstrip and size_chr > 1:
        return _rstrip_inner(chr_array, size_chr), len_chr, size_chr
    return chr_array, len_chr, size_chr


@register_jitable
def _register_array_strings(s, rstrip=True):
    """Expose the numerical representation of UTF-32 array strings."""
    len_chr = s.size
    size_chr = s.itemsize // 4
    chr_array = np.ravel(s).view(np.dtype('int32'))
    if rstrip and size_chr > 1:
        return _rstrip_inner(chr_array, size_chr), len_chr, size_chr
    return chr_array, len_chr, size_chr


@register_jitable
def _rstrip_inner(chr_array, size_chr):
    r"""
    Removes trailing \t\n\r\f\v\s characters.
    As is the case when used on character comparison operators, this variation
    ignores the first character.
    """
    if size_chr == 1:
        return chr_array

    def bisect_null(a, j, k):
        """Bisect null right-padded strings with the form '\x00'."""
        while j < k:
            m = (k + j) // 2
            c = a[m]
            if c != 0:
                j = m + 1
            elif c == 0:
                k = m
            else:
                return m
        return j

    whitespace = {0, 9, 10, 11, 12, 13, 32}
    size_stride = size_chr - 1

    for i in range(size_stride, chr_array.size, size_chr):
        if chr_array[i] not in whitespace:
            continue

        o = i - size_stride
        p = bisect_null(chr_array, o, i - 1)
        while p > o and chr_array[p] in whitespace:
            p -= 1
        chr_array[p + 1: i + 1] = 0

    return chr_array


# -----------------------------------------------------------------------------
# Comparison Operators


@register_jitable
def _cast_comparison(size_chr, len_chr, size_cmp, len_cmp):
    """
    Determines the character offsets used to align the comparison to the target.
    """
    if len_cmp > 1 and len_cmp != len_chr:
        raise ValueError('shape mismatch: objects cannot be broadcast to a '
                         'single shape.  Mismatch is between arg 0 and arg 1.')
    size_margin = size_chr - size_cmp
    if len_cmp == 1:
        size_stride = min(size_chr, size_cmp + (size_margin < 0))
        size_cmp = 0
    else:
        size_stride = min(size_chr, size_cmp)
    return size_cmp, size_stride, size_margin


@register_jitable(locals={'cmp_ord': types.int32})
def greater_equal(chr_array, len_chr, size_chr,
                  cmp_array, len_cmp, size_cmp, inv=False):
    """Native Implementation of np.char.greater_equal"""
    if 1 == size_chr == size_cmp:
        return cmp_array >= chr_array if inv else chr_array >= cmp_array

    size_cmp, size_stride, size_margin = _cast_comparison(size_chr, len_chr,
                                                          size_cmp, len_cmp)
    greater_equal_than = np.zeros(len_chr, 'bool')
    stride = stride_cmp = 0
    for i in range(len_chr):
        for j in range(size_stride):
            cmp_ord = chr_array[stride + j] - cmp_array[stride_cmp + j]
            if cmp_ord != 0:
                greater_equal_than[i] = ((inv and -cmp_ord) or cmp_ord) >= 0
                break
        else:
            greater_equal_than[i] = size_margin >= 0 \
                or not cmp_array[stride_cmp + size_stride]
        stride += size_chr
        stride_cmp += size_cmp
    return greater_equal_than


@register_jitable(locals={'cmp_ord': types.int32})
def greater(chr_array, len_chr, size_chr,
            cmp_array, len_cmp, size_cmp, inv=False):
    """Native Implementation of np.char.greater"""
    if 1 == size_chr == size_cmp:
        return cmp_array > chr_array if inv else chr_array > cmp_array

    size_cmp, size_stride, size_margin = _cast_comparison(size_chr, len_chr,
                                                          size_cmp, len_cmp)
    greater_than = np.zeros(len_chr, 'bool')
    stride = stride_cmp = 0
    for i in range(len_chr):
        for j in range(size_stride):
            cmp_ord = chr_array[stride + j] - cmp_array[stride_cmp + j]
            if cmp_ord != 0:
                greater_than[i] = ((inv and -cmp_ord) or cmp_ord) >= 0
                break
        else:
            greater_than[i] = size_margin > 0 \
                and chr_array[stride + size_stride]
        stride += size_chr
        stride_cmp += size_cmp
    return greater_than


@register_jitable
def equal(chr_array, len_chr, size_chr, cmp_array, len_cmp, size_cmp):
    """Native Implementation of np.char.equal"""
    if 1 == size_chr == size_cmp:
        return chr_array == cmp_array

    if 1 == len_cmp and size_chr < size_cmp and cmp_array[size_chr]:
        return np.zeros(len_chr, 'bool')

    size_cmp, size_stride, size_margin = _cast_comparison(size_chr, len_chr,
                                                          size_cmp, len_cmp)
    equal_to = np.zeros(len_chr, 'bool')
    stride = stride_cmp = 0
    for i in range(len_chr):
        for j in range(size_stride):
            if chr_array[stride + j] != cmp_array[stride_cmp + j]:
                break
        else:
            equal_to[i] = (
                    not size_margin
                    or (size_margin > 0 and not chr_array[stride + size_stride])
                    or (size_margin < 0
                        and not cmp_array[stride_cmp + size_stride])
            )
        stride += size_chr
        stride_cmp += size_cmp
    return equal_to


@register_jitable
def compare_chararrays(chr_array, len_chr, size_chr,
                       cmp_array, len_cmp, size_cmp, inv, cmp):
    """Native Implementation of np.char.compare_chararrays"""
    # {“<”, “<=”, “==”, “>=”, “>”, “!=”}
    # { (60,) (60, 61), (61, 61), (62, 61), (62,), (33, 61) }
    # The argument cmp can be passed as bytes or string, hence ordinal mapping.
    if len(cmp) == 1:
        cmp_ord = ord(cmp)
        if cmp_ord == 60:
            return ~greater_equal(chr_array, len_chr, size_chr,
                                  cmp_array, len_cmp, size_cmp, inv)
        elif cmp_ord == 62:
            return greater(chr_array, len_chr, size_chr,
                           cmp_array, len_cmp, size_cmp, inv)
    elif len(cmp) == 2 and ord(cmp[1]) == 61:
        cmp_ord = ord(cmp[0])
        if cmp_ord == 60:
            return ~greater(chr_array, len_chr, size_chr,
                            cmp_array, len_cmp, size_cmp, inv)
        elif cmp_ord == 61:
            return equal(chr_array, len_chr, size_chr,
                         cmp_array, len_cmp, size_cmp)
        elif cmp_ord == 62:
            return greater_equal(chr_array, len_chr, size_chr,
                                 cmp_array, len_cmp, size_cmp, inv)
        elif cmp_ord == 33:
            return ~equal(chr_array, len_chr, size_chr,
                          cmp_array, len_cmp, size_cmp)
    raise ValueError("comparison must be '==', '!=', '<', '>', '<=', '>='")


def _ensure_type(x):
    ndim = -1
    if isinstance(x, types.Array):
        ndim = x.ndim
        if ndim > 1 or x.layout != 'C':
            msg = 'shape mismatch: objects cannot be broadcast to a single ' \
                  'shape.  Mismatch is between arg 0 and arg 1.'
            raise ValueError(msg)
        x = x.dtype
        if not isinstance(x, (types.CharSeq,
                              types.UnicodeCharSeq)) or not x.count:
            raise TypeError('comparison of non-string arrays')
    elif not isinstance(x, (types.Bytes, types.UnicodeType)):
        raise TypeError('comparison of non-string arrays')
    return x, ndim


def _get_register_type(x1, x2):
    """Determines the call function for the comparison pair based on type."""
    (x1_type, x1_dim), (x2_type, x2_dim) = _ensure_type(x1), _ensure_type(x2)
    byte_types = (types.Bytes, types.CharSeq)
    str_types = (types.UnicodeType, types.UnicodeCharSeq)

    if isinstance(x1_type, byte_types) and isinstance(x2_type, byte_types):
        register_x1 = _register_array_bytes if x1_dim >= 0 \
            else _register_scalar_bytes
        register_x2 = _register_array_bytes if x2_dim >= 0 \
            else _register_scalar_bytes
    elif isinstance(x1_type, str_types) and isinstance(x2_type, str_types):
        register_x1 = _register_array_strings if x1_dim >= 0 \
            else _register_scalar_strings
        register_x2 = _register_array_strings if x2_dim >= 0 \
            else _register_scalar_strings
    else:
        raise NotImplementedError('NotImplemented')
    return register_x1, register_x2, x1_dim, x2_dim


@overload(np.char.equal)
def ov_char_equal(x1, x2):
    register_x1, register_x2, x1_dim, x2_dim = _get_register_type(x1, x2)

    if x1_dim > 0 or x2_dim > 0:
        def impl(x1, x2):
            if x1_dim < 0 <= x2_dim:
                return equal(*register_x2(x2), *register_x1(x1))
            return equal(*register_x1(x1), *register_x2(x2))
    else:
        def impl(x1, x2):
            return np.array(equal(*register_x1(x1), *register_x2(x2))[0])
    return impl


@overload(np.char.not_equal)
def ov_char_not_equal(x1, x2):
    register_x1, register_x2, x1_dim, x2_dim = _get_register_type(x1, x2)

    if x1_dim > 0 or x2_dim > 0:
        def impl(x1, x2):
            if x1_dim < 0 <= x2_dim:
                return ~equal(*register_x2(x2), *register_x1(x1))
            return ~equal(*register_x1(x1), *register_x2(x2))
    else:
        def impl(x1, x2):
            return np.array(~equal(*register_x1(x1), *register_x2(x2))[0])
    return impl


@overload(np.char.greater_equal)
def ov_char_greater_equal(x1, x2):
    register_x1, register_x2, x1_dim, x2_dim = _get_register_type(x1, x2)

    if x1_dim > 0 or x2_dim > 0:
        def impl(x1, x2):
            if x1_dim < 0 <= x2_dim:
                return greater_equal(*register_x2(x2), *register_x1(x1), True)
            return greater_equal(*register_x1(x1), *register_x2(x2))
    else:
        def impl(x1, x2):
            return np.array(greater_equal(*register_x1(x1),
                                          *register_x2(x2))[0])
    return impl


@overload(np.char.greater)
def ov_char_greater(x1, x2):
    register_x1, register_x2, x1_dim, x2_dim = _get_register_type(x1, x2)

    if x1_dim > 0 or x2_dim > 0:
        def impl(x1, x2):
            if x1_dim < 0 <= x2_dim:
                return greater(*register_x2(x2), *register_x1(x1), True)
            return greater(*register_x1(x1), *register_x2(x2))
    else:
        def impl(x1, x2):
            return np.array(greater(*register_x1(x1), *register_x2(x2))[0])
    return impl


@overload(np.char.less)
def ov_char_less(x1, x2):
    register_x1, register_x2, x1_dim, x2_dim = _get_register_type(x1, x2)

    if x1_dim > 0 or x2_dim > 0:
        def impl(x1, x2):
            if x1_dim < 0 <= x2_dim:
                return ~greater_equal(*register_x2(x2), *register_x1(x1), True)
            return ~greater_equal(*register_x1(x1), *register_x2(x2))
    else:
        def impl(x1, x2):
            return np.array(~greater_equal(*register_x1(x1),
                                           *register_x2(x2))[0])
    return impl


@overload(np.char.less_equal)
def ov_char_less_equal(x1, x2):
    register_x1, register_x2, x1_dim, x2_dim = _get_register_type(x1, x2)

    if x1_dim > 0 or x2_dim > 0:
        def impl(x1, x2):
            if x1_dim < 0 <= x2_dim:
                return ~greater(*register_x2(x2), *register_x1(x1), True)
            return ~greater(*register_x1(x1), *register_x2(x2))
    else:
        def impl(x1, x2):
            return np.array(~greater(*register_x1(x1), *register_x2(x2))[0])
    return impl


@overload(np.char.compare_chararrays)
def ov_char_compare_chararrays(a1, a2, cmp, rstrip):
    if not isinstance(cmp, (types.Bytes, types.UnicodeType)):
        raise TypeError(f'a bytes-like object is required, not {cmp.name}')

    register_a1, register_a2, a1_dim, a2_dim = _get_register_type(a1, a2)

    if a1_dim > 0 or a2_dim > 0:
        def impl(a1, a2, cmp, rstrip):
            if a1_dim < 0 <= a2_dim:
                return compare_chararrays(*register_a2(a2, rstrip),
                                          *register_a1(a1, rstrip), True, cmp)
            return compare_chararrays(*register_a1(a1, rstrip),
                                      *register_a2(a2, rstrip), False, cmp)
    else:
        def impl(a1, a2, cmp, rstrip):
            return np.array(compare_chararrays(*register_a1(a1, rstrip),
                                               *register_a2(a2, rstrip),
                                               False, cmp)[0])
    return impl


# -----------------------------------------------------------------------------
