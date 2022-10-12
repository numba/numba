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
            _rstrip_inner(np.frombuffer(b, 'uint8').copy(), size_chr, True),
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
        return _rstrip_inner(chr_array, size_chr, True), len_chr, size_chr
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
def _rstrip_inner(chr_array, size_chr, is_scalar=False):
    r"""
    Removes trailing \t\n\r\f\v\s characters.
    As is the case when used on character comparison operators, this variation
    ignores the first character.
    """
    if size_chr == 1:
        return chr_array

    whitespace = {0, 9, 10, 11, 12, 13, 32}
    size_stride = size_chr - 1

    if is_scalar or size_chr < 9:
        for i in range(size_stride, chr_array.size, size_chr):
            for p in range(i, i - size_stride, -1):
                if chr_array[p] not in whitespace:
                    break
                chr_array[p] = 0
    else:
        for i in range(size_stride, chr_array.size, size_chr):
            if chr_array[i] in whitespace:
                o = i - size_stride
                p = bisect_null(chr_array, o, i - 1)
                while p > o and chr_array[p] in whitespace:
                    p -= 1
                chr_array[p + 1: i + 1] = 0
    return chr_array


@register_jitable
def bisect_null(a, j, k):
    """Bisect null right-padded strings with the form '\x00'."""
    while j < k:
        m = (k + j) // 2
        if a[m]:
            j = m + 1
        else:
            k = m
    return j


def _ensure_type(x, exception: Exception = None):
    """Ensure argument is a character type with appropriate layout and shape."""
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
            ndim = None
    elif not isinstance(x, (types.Bytes, types.UnicodeType)):
        ndim = None
    if isinstance(exception, Exception) and ndim is None:
        raise exception
    return x, ndim


def _str_type(x, as_np=True):
    """Infer string-type of an objects Numba instance."""
    if isinstance(x, types.Array):
        if isinstance(x.dtype, types.CharSeq):
            return 'numpy.bytes_' if as_np else 'bytes'
        if isinstance(x.dtype, types.UnicodeCharSeq):
            return 'numpy.str_' if as_np else 'str'
        return f'like {x.dtype.name}'

    if isinstance(x, types.Bytes):
        return 'numpy.bytes_' if as_np else 'bytes'
    if isinstance(x, types.UnicodeType):
        return 'numpy.str_' if as_np else 'str'
    return f'like {x.name}'


def _register_pair(x1, x2, exception: (Exception, int) = None):
    """Determines the call function for the comparison pair, based on type."""
    e = exception or TypeError("comparison of non-string arrays")
    x1_type, x1_dim = _ensure_type(x1, e)
    x2_type, x2_dim = _ensure_type(x2, e)

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
        if exception == 1:
            as_type = _str_type(x1, as_np=False)
            if as_type in ('str', 'bytes'):
                e = TypeError(f"must be {as_type}, not {_str_type(x2)}")
            else:
                e = TypeError("string operation on non-string array")
        else:
            e = NotImplementedError('NotImplemented')
        raise e
    return register_x1, register_x2, x1_dim, x2_dim


def _register_single(x1, exception: Exception = None):
    """Determines the call function for the input, based on type."""
    e = exception or TypeError("string operation on non-string array")
    x1_type, x1_dim = _ensure_type(x1, e)

    byte_types = (types.Bytes, types.CharSeq)
    str_types = (types.UnicodeType, types.UnicodeCharSeq)

    if isinstance(x1_type, byte_types):
        register_x1 = _register_array_bytes if x1_dim >= 0 \
            else _register_scalar_bytes
        as_bytes = True
    elif isinstance(x1_type, str_types):
        register_x1 = _register_array_bytes if x1_dim >= 0 \
            else _register_scalar_strings
        as_bytes = False
    else:
        e = exception or TypeError("string operation on non-string array")
        raise e
    return register_x1, x1_dim, as_bytes


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
    greater_equal_than = np.empty(len_chr, 'bool')
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
    greater_than = np.empty(len_chr, 'bool')
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


@overload(np.char.equal)
def ov_char_equal(x1, x2):
    register_x1, register_x2, x1_dim, x2_dim = _register_pair(x1, x2)

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
    register_x1, register_x2, x1_dim, x2_dim = _register_pair(x1, x2)

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
    register_x1, register_x2, x1_dim, x2_dim = _register_pair(x1, x2)

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
    register_x1, register_x2, x1_dim, x2_dim = _register_pair(x1, x2)

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
    register_x1, register_x2, x1_dim, x2_dim = _register_pair(x1, x2)

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
    register_x1, register_x2, x1_dim, x2_dim = _register_pair(x1, x2)

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

    register_a1, register_a2, a1_dim, a2_dim = _register_pair(a1, a2)

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
# String Information


def _ensure_slice(start, end):
    """Ensure start and end slice argument is an integer type."""
    slice_types = (types.Integer, types.NoneType)
    if not (isinstance(start, slice_types) and isinstance(end, slice_types)):
        raise TypeError("slice indices must be integers or None "
                        "or have an __index__ method")
    return 0, np.iinfo(np.int64).max


@register_jitable
def _init_sub_indices(start, end, size_chr):
    """Initialize substring start and end indices"""
    if end is None:
        end = size_chr
    else:
        end = max(min(end, size_chr), -size_chr)
    if start < 0:
        start = max(start, -size_chr)
    return start, end


@register_jitable
def _get_sub_indices(chr_lens, len_chr,
                     sub_lens, len_sub,
                     start, end, i):
    """Calculate substring start and end indices"""
    n_chr = chr_lens[(len_chr > 1 and i) or 0]
    n_sub = sub_lens[(len_sub > 1 and i) or 0]
    o = max(start < 0 and start + n_chr or start, 0)
    n = min(n_chr, max(end < 0 and end + n_chr or end, 0))
    return n_chr, n_sub, o, n


@register_jitable(locals={'end': types.int64})
def count(chr_array, len_chr, size_chr,
          sub_array, len_sub, size_sub, start, end):
    """Native Implementation of np.char.count"""

    start, end = _init_sub_indices(start, end, size_chr)
    if start > size_chr or start > end + size_chr:
        return np.zeros(max(len_chr, len_sub), 'int64')

    chr_lens = str_len(chr_array, len_chr, size_chr)
    sub_lens = str_len(sub_array, len_sub, size_sub)

    len_cast = max(len_chr, len_sub)
    count_sub = np.zeros(len_cast, 'int64')

    size_chr = (len_chr > 1 and size_chr) or 0
    size_cmp = (len_sub > 1 and size_sub) or 0
    stride = stride_cmp = 0
    for i in range(len_cast):
        n_chr, n_sub, o, n = _get_sub_indices(chr_lens, len_chr,
                                              sub_lens, len_sub,
                                              start, end, i)
        if n_sub:
            o += stride
            n += stride
            while o + n_sub <= n:
                for p in range(n_sub):
                    if chr_array[o + p] != sub_array[stride_cmp + p]:
                        o += 1
                        break
                else:
                    count_sub[i] += 1
                    o += n_sub
        else:
            count_sub[i] = o <= n and max(1 + n - o, 1)
        stride += size_chr
        stride_cmp += size_cmp
    return count_sub


@register_jitable
def endswith(chr_array, len_chr, size_chr,
             sub_array, len_sub, size_sub,
             start, end):
    """Native Implementation of np.char.endswith"""
    start, end = _init_sub_indices(start, end, size_chr)
    if start > size_chr or start > end + size_chr:
        return np.zeros(max(len_chr, len_sub), 'bool')

    chr_lens = str_len(chr_array, len_chr, size_chr)
    sub_lens = str_len(sub_array, len_sub, size_sub)

    len_cast = max(len_chr, len_sub)
    endswith_sub = np.ones(len_cast, 'bool')

    size_chr = (len_chr > 1 and size_chr) or 0
    size_cmp = (len_sub > 1 and size_sub) or 0
    stride = stride_cmp = 0
    for i in range(len_cast):
        n_chr, n_sub, o, n = _get_sub_indices(chr_lens, len_chr,
                                              sub_lens, len_sub,
                                              start, end, i)
        if o + n_sub <= n:
            n += stride - 1
            r = stride_cmp + n_sub - 1
            for p in range(n_sub):
                if chr_array[n - p] != sub_array[r - p]:
                    endswith_sub[i] = False
                    break
        else:
            endswith_sub[i] = not n_sub and o <= n
        stride += size_chr
        stride_cmp += size_cmp
    return endswith_sub


@register_jitable
def startswith(chr_array, len_chr, size_chr,
               sub_array, len_sub, size_sub,
               start, end):
    """Native Implementation of np.char.startswith"""
    start, end = _init_sub_indices(start, end, size_chr)
    if start > size_chr or start > end + size_chr:
        return np.zeros(max(len_chr, len_sub), 'bool')

    chr_lens = str_len(chr_array, len_chr, size_chr)
    sub_lens = str_len(sub_array, len_sub, size_sub)

    len_cast = max(len_chr, len_sub)
    startswith_sub = np.ones(len_cast, 'bool')

    size_chr = (len_chr > 1 and size_chr) or 0
    size_cmp = (len_sub > 1 and size_sub) or 0
    stride = stride_cmp = 0
    for i in range(len_cast):
        n_chr, n_sub, o, n = _get_sub_indices(chr_lens, len_chr,
                                              sub_lens, len_sub,
                                              start, end, i)
        if o + n_sub <= n:
            o = stride + o
            for p in range(n_sub):
                if chr_array[o + p] != sub_array[stride_cmp + p]:
                    startswith_sub[i] = False
                    break
        else:
            startswith_sub[i] = not n_sub and o <= n
        stride += size_chr
        stride_cmp += size_cmp
    return startswith_sub


@register_jitable
def find(chr_array, len_chr, size_chr,
         sub_array, len_sub, size_sub, start, end):
    """Native Implementation of np.char.find"""

    start, end = _init_sub_indices(start, end, size_chr)
    if start > size_chr or start > end + size_chr:
        return -np.ones(max(len_chr, len_sub), 'int64')

    chr_lens = str_len(chr_array, len_chr, size_chr)
    sub_lens = str_len(sub_array, len_sub, size_sub)

    len_cast = max(len_chr, len_sub)
    find_sub = -np.ones(len_cast, 'int64')

    size_chr = (len_chr > 1 and size_chr) or 0
    size_cmp = (len_sub > 1 and size_sub) or 0
    stride = stride_cmp = 0
    for i in range(len_cast):
        n_chr, n_sub, o, n = _get_sub_indices(chr_lens, len_chr,
                                              sub_lens, len_sub,
                                              start, end, i)
        if n_sub:
            o += stride
            n += stride
            while o + n_sub <= n:
                for p in range(n_sub):
                    if chr_array[o + p] != sub_array[stride_cmp + p]:
                        o += 1
                        break
                else:
                    find_sub[i] = o - stride
                    break
        else:
            find_sub[i] = (o <= n and o + 1) - 1
        stride += size_chr
        stride_cmp += size_cmp
    return find_sub


@register_jitable
def index(chr_array, len_chr, size_chr,
          sub_array, len_sub, size_sub,
          start, end):
    """Native Implementation of np.char.index"""

    start, end = _init_sub_indices(start, end, size_chr)
    if start > size_chr or start > end + size_chr:
        raise ValueError('substring not found')

    chr_lens = str_len(chr_array, len_chr, size_chr)
    sub_lens = str_len(sub_array, len_sub, size_sub)

    len_cast = max(len_chr, len_sub)
    index_sub = -np.ones(len_cast, 'int64')

    size_chr = (len_chr > 1 and size_chr) or 0
    size_cmp = (len_sub > 1 and size_sub) or 0
    stride = stride_cmp = 0
    for i in range(len_cast):
        n_chr, n_sub, o, n = _get_sub_indices(chr_lens, len_chr,
                                              sub_lens, len_sub,
                                              start, end, i)
        if n_sub:
            o += stride
            n += stride
            while o + n_sub <= n:
                for p in range(n_sub):
                    if chr_array[o + p] != sub_array[stride_cmp + p]:
                        o += 1
                        break
                else:
                    index_sub[i] = o - stride
                    break
            else:
                raise ValueError('substring not found')
        else:
            if o > n:
                raise ValueError('substring not found')
            index_sub[i] = o
        stride += size_chr
        stride_cmp += size_cmp
    return index_sub


@register_jitable
def rfind(chr_array, len_chr, size_chr,
          sub_array, len_sub, size_sub,
          start, end):
    """Native Implementation of np.char.rfind"""

    start, end = _init_sub_indices(start, end, size_chr)
    if start > size_chr or start > end + size_chr:
        return -np.ones(max(len_chr, len_sub), 'int64')

    chr_lens = str_len(chr_array, len_chr, size_chr)
    sub_lens = str_len(sub_array, len_sub, size_sub)

    len_cast = max(len_chr, len_sub)
    rfind_sub = -np.ones(len_cast, 'int64')

    size_chr = (len_chr > 1 and size_chr) or 0
    size_cmp = (len_sub > 1 and size_sub) or 0
    stride = stride_cmp = 0
    for i in range(len_cast):
        n_chr, n_sub, o, n = _get_sub_indices(chr_lens, len_chr,
                                              sub_lens, len_sub,
                                              start, end, i)
        if n_sub:
            o += stride - 1
            n += stride - 1
            r = stride_cmp + n_sub - 1
            while n - n_sub >= o:
                for p in range(n_sub):
                    if chr_array[n - p] != sub_array[r - p]:
                        n -= 1
                        break
                else:
                    rfind_sub[i] = n - n_sub - stride + 1
                    break
        else:
            rfind_sub[i] = (o <= n and n + 1) - 1
        stride += size_chr
        stride_cmp += size_cmp
    return rfind_sub


@register_jitable
def rindex(chr_array, len_chr, size_chr,
           sub_array, len_sub, size_sub,
           start, end):
    """Native Implementation of np.char.rindex"""

    start, end = _init_sub_indices(start, end, size_chr)
    if start > size_chr or start > end + size_chr:
        raise ValueError('substring not found')

    chr_lens = str_len(chr_array, len_chr, size_chr)
    sub_lens = str_len(sub_array, len_sub, size_sub)

    len_cast = max(len_chr, len_sub)
    rfind_sub = -np.ones(len_cast, 'int64')

    size_chr = (len_chr > 1 and size_chr) or 0
    size_cmp = (len_sub > 1 and size_sub) or 0
    stride = stride_cmp = 0
    for i in range(len_cast):
        n_chr, n_sub, o, n = _get_sub_indices(chr_lens, len_chr,
                                              sub_lens, len_sub,
                                              start, end, i)
        if n_sub:
            o += stride - 1
            n += stride - 1
            r = stride_cmp + n_sub - 1
            while n - n_sub >= o:
                for p in range(n_sub):
                    if chr_array[n - p] != sub_array[r - p]:
                        n -= 1
                        break
                else:
                    rfind_sub[i] = n - n_sub - stride + 1
                    break
            else:
                raise ValueError('substring not found')
        else:
            if o > n:
                raise ValueError('substring not found')
            rfind_sub[i] = n
        stride += size_chr
        stride_cmp += size_cmp
    return rfind_sub


@register_jitable
def str_len(chr_array, len_chr, size_chr):
    """Native Implementation of np.char.str_len"""
    if not size_chr:
        return np.zeros(len_chr, 'int64')

    str_length = np.empty(len_chr, 'int64')
    stride = size_chr - 1
    j = 0
    for i in range(0, chr_array.size, size_chr):
        str_length[j] = (chr_array[i + stride] and size_chr) \
            or bisect_null(chr_array, i, i + stride) - i
        j += 1
    return str_length


@register_jitable
def isalpha(chr_array, len_chr, size_chr):
    """Native Implementation of np.char.isalpha"""
    # Restricted to extended ASCII range(0, 255).
    # Complete ordinal set requires length of N > 125,000.
    if not size_chr:
        return np.zeros(len_chr, 'bool')

    alpha = {
        65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82,
        83, 84, 85, 86, 87, 88, 89, 90, 97, 98, 99, 100, 101, 102, 103, 104,
        105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118,
        119, 120, 121, 122, 170, 181, 186, 192, 193, 194, 195, 196, 197, 198,
        199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212,
        213, 214, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227,
        228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241,
        242, 243, 244, 245, 246, 248, 249, 250, 251, 252, 253, 254
    }

    chr_lens = str_len(chr_array, len_chr, size_chr)
    is_alpha = np.zeros(len_chr, 'bool')
    stride = 0
    for i in range(len_chr):
        for c in range(chr_lens[i]):
            chr_ord = chr_array[stride + c]
            if chr_ord not in alpha:
                is_alpha[i] = False
                break
            is_alpha[i] |= chr_ord in alpha
        stride += size_chr
    return is_alpha


@register_jitable
def isalnum(chr_array, len_chr, size_chr):
    """Native Implementation of np.char.isalnum"""
    # Restricted to extended ASCII range(0, 255).
    # Complete ordinal set requires length of N > 127,000.
    if not size_chr:
        return np.zeros(len_chr, 'bool')

    alnum = {
        48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 65, 66, 67, 68, 69, 70, 71, 72,
        73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90,
        97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111,
        112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 170, 178, 179,
        181, 185, 186, 188, 189, 190, 192, 193, 194, 195, 196, 197, 198, 199,
        200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213,
        214, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228,
        229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242,
        243, 244, 245, 246, 248, 249, 250, 251, 252, 253, 254
    }

    chr_lens = str_len(chr_array, len_chr, size_chr)
    is_alnum = np.zeros(len_chr, 'bool')
    stride = 0
    for i in range(len_chr):
        for c in range(chr_lens[i]):
            chr_ord = chr_array[stride + c]
            if chr_ord not in alnum:
                is_alnum[i] = False
                break
            is_alnum[i] |= chr_ord in alnum
        stride += size_chr
    return is_alnum


@register_jitable
def isdecimal(chr_array, len_chr, size_chr):
    """Native Implementation of np.char.isdecimal"""
    # Restricted to extended ASCII range(0, 255).
    # Complete ordinal set requires length of N > 600.
    if not size_chr:
        return np.zeros(len_chr, 'bool')

    decimal = {48, 49, 50, 51, 52, 53, 54, 55, 56, 57}

    chr_lens = str_len(chr_array, len_chr, size_chr)
    is_decimal = np.zeros(len_chr, 'bool')
    stride = 0
    for i in range(len_chr):
        for c in range(chr_lens[i]):
            chr_ord = chr_array[stride + c]
            if chr_ord not in decimal:
                is_decimal[i] = False
                break
            is_decimal[i] |= chr_ord in decimal
        stride += size_chr
    return is_decimal


@register_jitable
def isdigit(chr_array, len_chr, size_chr):
    """Native Implementation of np.char.isdigit"""
    # Restricted to extended ASCII range(0, 255).
    # Complete ordinal set requires length of N > 700.
    if not size_chr:
        return np.zeros(len_chr, 'bool')

    digit = {48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 178, 179, 185}

    chr_lens = str_len(chr_array, len_chr, size_chr)
    is_digit = np.zeros(len_chr, 'bool')
    stride = 0
    for i in range(len_chr):
        for c in range(chr_lens[i]):
            chr_ord = chr_array[stride + c]
            if chr_ord not in digit:
                is_digit[i] = False
                break
            is_digit[i] |= chr_ord in digit
        stride += size_chr
    return is_digit


@register_jitable
def isnumeric(chr_array, len_chr, size_chr):
    """Native Implementation of np.char.isnumeric"""
    # Restricted to extended ASCII range(0, 255).
    # Complete ordinal set requires length of N > 1800.
    if not size_chr:
        return np.zeros(len_chr, 'bool')

    numeric = {
        48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 178, 179, 185, 188, 189, 190
    }

    chr_lens = str_len(chr_array, len_chr, size_chr)
    is_numeric = np.zeros(len_chr, 'bool')
    stride = 0
    for i in range(len_chr):
        for c in range(chr_lens[i]):
            chr_ord = chr_array[stride + c]
            if chr_ord not in numeric:
                is_numeric[i] = False
                break
            is_numeric[i] |= chr_ord in numeric
        stride += size_chr
    return is_numeric


@register_jitable
def isspace(chr_array, len_chr, size_chr, as_bytes):
    """Native Implementation of np.char.isspace"""
    if not size_chr:
        return np.zeros(len_chr, 'bool')

    if as_bytes:
        space = {9, 10, 11, 12, 13, 32}
    else:
        space = {
            9, 10, 11, 12, 13, 28, 29, 30, 31, 32, 133, 160, 5760, 8192, 8193,
            8194, 8195, 8196, 8197, 8198, 8199, 8200, 8201, 8202, 8232, 8233,
            8239, 8287, 12288
        }

    chr_lens = str_len(chr_array, len_chr, size_chr)
    is_space = np.zeros(len_chr, 'bool')
    stride = 0
    for i in range(len_chr):
        for c in range(chr_lens[i]):
            chr_ord = chr_array[stride + c]
            if chr_ord not in space:
                is_space[i] = False
                break
            is_space[i] |= chr_ord in space
        stride += size_chr
    return is_space


@register_jitable
def istitle(chr_array, len_chr, size_chr):
    """Native Implementation of np.char.istitle"""
    # Restricted to extended ASCII range(0, 255).
    # Complete ordinal set requires length of N > 1900.
    if not size_chr:
        return np.zeros(len_chr, 'bool')

    lower = {
        97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111,
        112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 170, 181, 186,
        223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236,
        237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 248, 249, 250, 251,
        252, 253, 254
    }

    upper = {
        65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82,
        83, 84, 85, 86, 87, 88, 89, 90, 192, 193, 194, 195, 196, 197, 198, 199,
        200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213,
        214, 216, 217, 218, 219, 220, 221, 222
    }

    chr_lens = str_len(chr_array, len_chr, size_chr)
    is_title = np.zeros(len_chr, 'bool')
    stride = 0
    for i in range(len_chr):
        cased_state = False
        for c in range(chr_lens[i]):
            chr_ord = chr_array[stride + c]
            if (cased_state and chr_ord in upper) \
                    or (not cased_state and chr_ord in lower):
                is_title[i] = False
                break
            cased_state = chr_ord in upper
            is_title[i] |= cased_state
            cased_state |= chr_ord in lower
        stride += size_chr
    return is_title


@register_jitable
def isupper(chr_array, len_chr, size_chr):
    """Native Implementation of np.char.isupper"""
    # Restricted to extended ASCII range(0, 255).
    # Complete ordinal set requires length of N > 1900.
    if not size_chr:
        return np.zeros(len_chr, 'bool')

    lower = {
        97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111,
        112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 170, 181, 186,
        223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236,
        237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 248, 249, 250, 251,
        252, 253, 254
    }

    upper = {
        65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82,
        83, 84, 85, 86, 87, 88, 89, 90, 192, 193, 194, 195, 196, 197, 198, 199,
        200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213,
        214, 216, 217, 218, 219, 220, 221, 222
    }

    chr_lens = str_len(chr_array, len_chr, size_chr)
    is_upper = np.zeros(len_chr, 'bool')
    stride = 0
    for i in range(len_chr):
        for c in range(chr_lens[i]):
            chr_ord = chr_array[stride + c]
            if chr_ord in lower:
                is_upper[i] = False
                break
            is_upper[i] |= chr_ord in upper
        stride += size_chr
    return is_upper


@register_jitable
def islower(chr_array, len_chr, size_chr):
    """Native Implementation of np.char.islower"""
    # Restricted to extended ASCII range(0, 255).
    # Complete ordinal set requires length of N > 2300.
    if not size_chr:
        return np.zeros(len_chr, 'bool')

    lower = {
        97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111,
        112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 170, 181, 186,
        223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236,
        237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 248, 249, 250, 251,
        252, 253, 254
    }

    upper = {
        65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82,
        83, 84, 85, 86, 87, 88, 89, 90, 192, 193, 194, 195, 196, 197, 198, 199,
        200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213,
        214, 216, 217, 218, 219, 220, 221, 222
    }

    chr_lens = str_len(chr_array, len_chr, size_chr)
    is_lower = np.zeros(len_chr, 'bool')
    stride = 0
    for i in range(len_chr):
        for c in range(chr_lens[i]):
            chr_ord = chr_array[stride + c]
            if chr_ord in upper:
                is_lower[i] = False
                break
            is_lower[i] |= chr_ord in lower
        stride += size_chr
    return is_lower


@overload(np.char.count)
def ov_char_count(a, sub, start=0, end=None):
    register_a, register_sub, a_dim, sub_dim = _register_pair(a, sub, 1)
    s, e = _ensure_slice(start, end)

    if a_dim > 0 or sub_dim > 0:
        def impl(a, sub, start=0, end=None):
            start = start or s
            end = e if end is None else end
            return count(*register_a(a, False),
                         *register_sub(sub, False),
                         start, end)
    else:
        def impl(a, sub, start=0, end=None):
            start = start or s
            end = e if end is None else end
            return np.array(count(*register_a(a, False),
                                  *register_sub(sub, False),
                                  start, end)[0])
    return impl


@overload(np.char.endswith)
def ov_char_endswith(a, sub, start=0, end=None):
    register_a, register_sub, a_dim, sub_dim = _register_pair(a, sub, 1)
    s, e = _ensure_slice(start, end)

    if a_dim > 0 or sub_dim > 0:
        def impl(a, sub, start=0, end=None):
            start = start or s
            end = e if end is None else end
            return endswith(*register_a(a, False),
                            *register_sub(sub, False),
                            start, end)
    else:
        def impl(a, sub, start=0, end=None):
            start = start or s
            end = e if end is None else end
            return np.array(endswith(*register_a(a, False),
                                     *register_sub(sub, False),
                                     start, end)[0])
    return impl


@overload(np.char.startswith)
def ov_char_startswith(a, sub, start=0, end=None):
    register_a, register_sub, a_dim, sub_dim = _register_pair(a, sub, 1)
    s, e = _ensure_slice(start, end)

    if a_dim > 0 or sub_dim > 0:
        def impl(a, sub, start=0, end=None):
            start = start or s
            end = e if end is None else end
            return startswith(*register_a(a, False),
                              *register_sub(sub, False),
                              start, end)
    else:
        def impl(a, sub, start=0, end=None):
            start = start or s
            end = e if end is None else end
            return np.array(startswith(*register_a(a, False),
                                       *register_sub(sub, False),
                                       start, end)[0])
    return impl


@overload(np.char.find)
def ov_char_find(a, sub, start=0, end=None):
    register_a, register_sub, a_dim, sub_dim = _register_pair(a, sub, 1)
    s, e = _ensure_slice(start, end)

    if a_dim > 0 or sub_dim > 0:
        def impl(a, sub, start=0, end=None):
            start = start or s
            end = e if end is None else end
            return find(*register_a(a, False),
                        *register_sub(sub, False),
                        start, end)
    else:
        def impl(a, sub, start=0, end=None):
            start = start or s
            end = e if end is None else end
            return np.array(find(*register_a(a, False),
                                 *register_sub(sub, False),
                                 start, end)[0])
    return impl


@overload(np.char.rfind)
def ov_char_rfind(a, sub, start=0, end=None):
    register_a, register_sub, a_dim, sub_dim = _register_pair(a, sub, 1)
    s, e = _ensure_slice(start, end)

    if a_dim > 0 or sub_dim > 0:
        def impl(a, sub, start=0, end=None):
            start = start or s
            end = e if end is None else end
            return rfind(*register_a(a, False),
                         *register_sub(sub, False),
                         start, end)
    else:
        def impl(a, sub, start=0, end=None):
            start = start or s
            end = e if end is None else end
            return np.array(rfind(*register_a(a, False),
                                  *register_sub(sub, False),
                                  start, end)[0])
    return impl


@overload(np.char.index)
def ov_char_index(a, sub, start=0, end=None):
    register_a, register_sub, a_dim, sub_dim = _register_pair(a, sub, 1)
    s, e = _ensure_slice(start, end)

    if a_dim > 0 or sub_dim > 0:
        def impl(a, sub, start=0, end=None):
            start = start or s
            end = e if end is None else end
            return index(*register_a(a, False),
                         *register_sub(sub, False),
                         start, end)
    else:
        def impl(a, sub, start=0, end=None):
            start = start or s
            end = e if end is None else end
            return np.array(index(*register_a(a, False),
                                  *register_sub(sub, False),
                                  start, end)[0])
    return impl


@overload(np.char.rindex)
def ov_char_rindex(a, sub, start=0, end=None):
    register_a, register_sub, a_dim, sub_dim = _register_pair(a, sub, 1)
    s, e = _ensure_slice(start, end)

    if a_dim > 0 or sub_dim > 0:
        def impl(a, sub, start=0, end=None):
            start = start or s
            end = e if end is None else end
            return rindex(*register_a(a, False),
                          *register_sub(sub, False),
                          start, end)
    else:
        def impl(a, sub, start=0, end=None):
            start = start or s
            end = e if end is None else end
            return np.array(rindex(*register_a(a, False),
                                   *register_sub(sub, False),
                                   start, end)[0])
    return impl


@overload(np.char.str_len)
def ov_str_len(a):
    register_a, a_dim, _ = _register_single(a)

    if a_dim > 0:
        def impl(a):
            return str_len(*register_a(a, False))
    else:
        def impl(a):
            return np.array(str_len(*register_a(a, False))[0])
    return impl


@overload(np.char.isalpha)
def ov_isalpha(a):
    register_a, a_dim, _ = _register_single(a)

    if a_dim > 0:
        def impl(a):
            return isalpha(*register_a(a, False))
    else:
        def impl(a):
            return np.array(isalpha(*register_a(a, False))[0])
    return impl


@overload(np.char.isalnum)
def ov_isalnum(a):
    register_a, a_dim, _ = _register_single(a)

    if a_dim > 0:
        def impl(a):
            return isalnum(*register_a(a, False))
    else:
        def impl(a):
            return np.array(isalnum(*register_a(a, False))[0])
    return impl


@overload(np.char.isspace)
def ov_isspace(a):
    register_a, a_dim, as_bytes = _register_single(a)

    if a_dim > 0:
        def impl(a):
            return isspace(*register_a(a, False), as_bytes)
    else:
        def impl(a):
            return np.array(isspace(*register_a(a, False), as_bytes)[0])
    return impl


@overload(np.char.isdecimal)
def ov_isdecimal(a):
    catch_incompatible = TypeError("isnumeric is only available for "
                                   "Unicode strings and arrays")
    register_a, a_dim, as_bytes = _register_single(a, catch_incompatible)
    if as_bytes:
        raise catch_incompatible

    if a_dim > 0:
        def impl(a):
            return isdecimal(*register_a(a, False))
    else:
        def impl(a):
            return np.array(isdecimal(*register_a(a, False))[0])
    return impl


@overload(np.char.isdigit)
def ov_isdigit(a):
    register_a, a_dim, _ = _register_single(a)

    if a_dim > 0:
        def impl(a):
            return isdigit(*register_a(a, False))
    else:
        def impl(a):
            return np.array(isdigit(*register_a(a, False))[0])
    return impl


@overload(np.char.isnumeric)
def ov_isnumeric(a):
    catch_incompatible = TypeError("isnumeric is only available for "
                                   "Unicode strings and arrays")
    register_a, a_dim, as_bytes = _register_single(a, catch_incompatible)
    if as_bytes:
        raise catch_incompatible

    if a_dim > 0:
        def impl(a):
            return isnumeric(*register_a(a, False))
    else:
        def impl(a):
            return np.array(isnumeric(*register_a(a, False))[0])
    return impl


@overload(np.char.istitle)
def ov_istitle(a):
    register_a, a_dim, _ = _register_single(a)

    if a_dim > 0:
        def impl(a):
            return istitle(*register_a(a, False))
    else:
        def impl(a):
            return np.array(istitle(*register_a(a, False))[0])
    return impl


@overload(np.char.isupper)
def ov_isupper(a):
    register_a, a_dim, _ = _register_single(a)

    if a_dim > 0:
        def impl(a):
            return isupper(*register_a(a, False))
    else:
        def impl(a):
            return np.array(isupper(*register_a(a, False))[0])
    return impl


@overload(np.char.islower)
def ov_islower(a):
    register_a, a_dim, _ = _register_single(a)

    if a_dim > 0:
        def impl(a):
            return islower(*register_a(a, False))
    else:
        def impl(a):
            return np.array(islower(*register_a(a, False))[0])
    return impl


# ------------------------------------------------------------------------------
