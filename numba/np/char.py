"""Implementation of numpy.char routines."""

from numba.extending import overload, register_jitable
from numba import types
import numpy as np


# -----------------------------------------------------------------------------
# Support Functions


@register_jitable
def _register_bytes(b, rstrip=True):
    """Expose the numerical representation of ASCII bytes."""
    if isinstance(b, bytes):
        len_chr = 1
        size_chr = len(b)
    else:
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
def _register_strings(s, rstrip=True):
    """Expose the numerical representation of UTF-32 strings."""
    if isinstance(s, str):
        len_chr = 1
        size_chr = len(s)
        chr_array = np.empty(size_chr, 'int32')
        for i in range(size_chr):
            chr_array[i] = ord(s[i])
    else:
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
def _cast_comparison(size_chr, len_chr, len_cmp, size_cmp):
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


@register_jitable
def _compare_any(x1: np.ndarray, x2: np.ndarray) -> bool:
    for i in range(x1.size):
        if x1[i] != x2[i]:
            return True
    return False


@register_jitable(locals={'cmp_ord': types.int32})
def greater_equal(chr_array, len_chr, size_chr,
                  cmp_array, len_cmp, size_cmp, inv=False):
    """Native Implementation of np.char.greater_equal"""
    if 1 == size_chr == size_cmp:
        return cmp_array >= chr_array if inv else chr_array >= cmp_array

    size_cmp, size_stride, size_margin = _cast_comparison(size_chr, len_chr,
                                                          len_cmp, size_cmp)
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
                                                          len_cmp, size_cmp)
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
    ix = 0
    if len_cmp == 1:
        if size_chr < size_cmp:
            return np.zeros(len_chr, 'bool')
        equal_to = np.empty(len_chr, 'bool')
        if size_chr > size_cmp:
            for i in range(len_chr):
                equal_to[i] = chr_array[ix + size_cmp] == 0 \
                              and not _compare_any(cmp_array,
                                                   chr_array[ix:ix + size_cmp])
                ix += size_chr
        else:
            for i in range(len_chr):
                equal_to[i] = not _compare_any(cmp_array,
                                               chr_array[ix:ix + size_cmp])
                ix += size_chr
    elif len_chr == len_cmp:
        iy = 0
        equal_to = np.empty(len_chr, 'bool')
        if size_chr < size_cmp:
            for i in range(len_chr):
                equal_to[i] = cmp_array[iy + size_chr] == 0 \
                              and not _compare_any(chr_array[ix:ix + size_chr],
                                                   cmp_array[iy:iy + size_chr])
                ix += size_chr
                iy += size_cmp
        elif size_chr > size_cmp:
            for i in range(len_chr):
                equal_to[i] = chr_array[ix + size_cmp] == 0 \
                              and not _compare_any(chr_array[ix:ix + size_cmp],
                                                   cmp_array[iy:iy + size_cmp])
                ix += size_chr
                iy += size_cmp
        else:
            if size_chr == 1:
                return chr_array == cmp_array
            for i in range(len_chr):
                equal_to[i] = not _compare_any(chr_array[ix:ix + size_chr],
                                               cmp_array[ix:ix + size_chr])
                ix += size_chr
    else:
        msg = 'shape mismatch: objects cannot be broadcast to a single shape.' \
              '  Mismatch is between arg 0 and arg 1.'
        raise ValueError(msg)
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


@register_jitable
def _ensure_type(x):
    ndim = 0
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


@register_jitable
def _get_register_type(x1, x2):

    (x1_type, x1_dim), (x2_type, x2_dim) = _ensure_type(x1), _ensure_type(x2)
    byte_types = (types.Bytes, types.CharSeq)
    str_types = (types.UnicodeType, types.UnicodeCharSeq)

    register_type = cmp_type = None
    if isinstance(x1_type, byte_types) and isinstance(x2_type, byte_types):
        register_type = _register_bytes
        cmp_type = bytes
    elif isinstance(x1_type, str_types) and isinstance(x2_type, str_types):
        register_type = _register_strings
        cmp_type = str

    if not register_type:
        raise NotImplementedError('NotImplemented')
    return register_type, cmp_type, x1_dim, x2_dim


@overload(np.char.equal)
def ov_char_equal(x1, x2):
    register_type, cmp_type, x1_dim, x2_dim = _get_register_type(x1, x2)

    if x1_dim or x2_dim:
        def impl(x1, x2):
            if isinstance(x1, cmp_type) and not isinstance(x2, cmp_type):
                return equal(*register_type(x2), *register_type(x1))
            return equal(*register_type(x1), *register_type(x2))
    else:
        def impl(x1, x2):
            return np.array(equal(*register_type(x1), *register_type(x2))[0])
    return impl


@overload(np.char.not_equal)
def ov_char_not_equal(x1, x2):
    register_type, cmp_type, x1_dim, x2_dim = _get_register_type(x1, x2)

    if x1_dim or x2_dim:
        def impl(x1, x2):
            if isinstance(x1, cmp_type) and not isinstance(x2, cmp_type):
                return ~equal(*register_type(x2), *register_type(x1))
            return ~equal(*register_type(x1), *register_type(x2))
    else:
        def impl(x1, x2):
            return np.array(~equal(*register_type(x1), *register_type(x2))[0])
    return impl


@overload(np.char.greater_equal)
def ov_char_greater_equal(x1, x2):
    register_type, cmp_type, x1_dim, x2_dim = _get_register_type(x1, x2)

    if x1_dim or x2_dim:
        def impl(x1, x2):
            if isinstance(x1, cmp_type) and not isinstance(x2, cmp_type):
                return greater_equal(*register_type(x2),
                                     *register_type(x1), True)
            return greater_equal(*register_type(x1), *register_type(x2))
    else:
        def impl(x1, x2):
            return np.array(greater_equal(*register_type(x1),
                                          *register_type(x2))[0])
    return impl


@overload(np.char.greater)
def ov_char_greater(x1, x2):
    register_type, cmp_type, x1_dim, x2_dim = _get_register_type(x1, x2)

    if x1_dim or x2_dim:
        def impl(x1, x2):
            if isinstance(x1, cmp_type) and not isinstance(x2, cmp_type):
                return greater(*register_type(x2), *register_type(x1), True)
            return greater(*register_type(x1), *register_type(x2))
    else:
        def impl(x1, x2):
            return np.array(greater(*register_type(x1), *register_type(x2))[0])
    return impl


@overload(np.char.less)
def ov_char_less(x1, x2):
    register_type, cmp_type, x1_dim, x2_dim = _get_register_type(x1, x2)

    if x1_dim or x2_dim:
        def impl(x1, x2):
            if isinstance(x1, cmp_type) and not isinstance(x2, cmp_type):
                return ~greater_equal(*register_type(x2),
                                      *register_type(x1), True)
            return ~greater_equal(*register_type(x1), *register_type(x2))
    else:
        def impl(x1, x2):
            return np.array(~greater_equal(*register_type(x1),
                                           *register_type(x2))[0])
    return impl


@overload(np.char.less_equal)
def ov_char_less_equal(x1, x2):
    register_type, cmp_type, x1_dim, x2_dim = _get_register_type(x1, x2)

    if x1_dim or x2_dim:
        def impl(x1, x2):
            if isinstance(x1, cmp_type) and not isinstance(x2, cmp_type):
                return ~greater(*register_type(x2), *register_type(x1), True)
            return ~greater(*register_type(x1), *register_type(x2))
    else:
        def impl(x1, x2):
            return np.array(~greater(*register_type(x1), *register_type(x2))[0])
    return impl


@register_jitable
@overload(np.char.compare_chararrays)
def ov_char_compare_chararrays(a1, a2, cmp, rstrip):
    if not isinstance(cmp, (types.Bytes, types.UnicodeType)):
        raise TypeError(f'a bytes-like object is required, not {cmp.name}')

    register_type, cmp_type, a1_dim, a2_dim = _get_register_type(a1, a2)

    if a1_dim or a2_dim:
        def impl(a1, a2, cmp, rstrip):
            if isinstance(a1, cmp_type) and not isinstance(a2, cmp_type):
                return compare_chararrays(*register_type(a2, rstrip),
                                          *register_type(a1, rstrip), True, cmp)
            return compare_chararrays(*register_type(a1, rstrip),
                                      *register_type(a2, rstrip), False, cmp)
    else:
        def impl(a1, a2, cmp, rstrip):
            return np.array(compare_chararrays(*register_type(a1, rstrip),
                                               *register_type(a2, rstrip),
                                               False, cmp)[0])
    return impl


# -----------------------------------------------------------------------------
