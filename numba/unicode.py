import operator

import numpy as np
from llvmlite.ir import IntType, Constant

from numba.extending import (
    models,
    register_model,
    make_attribute_wrapper,
    unbox,
    box,
    NativeValue,
    overload,
    overload_method,
    intrinsic,
)
from numba.targets.imputils import lower_constant, lower_cast
from numba import cgutils
from numba import types
from numba import njit
from numba.pythonapi import (
    PY_UNICODE_1BYTE_KIND,
    PY_UNICODE_2BYTE_KIND,
    PY_UNICODE_4BYTE_KIND,
    PY_UNICODE_WCHAR_KIND,
)
from numba.targets import slicing
from numba._helperlib import c_helpers
from numba.targets.hashing import _Py_hash_t
from numba.unsafe.bytes import memcpy_region

### DATA MODEL


@register_model(types.UnicodeType)
class UnicodeModel(models.StructModel):
    def __init__(self, dmm, fe_type):
        members = [
            ('data', types.voidptr),
            ('length', types.intp),
            ('kind', types.int32),
            ('hash', _Py_hash_t),
            ('meminfo', types.MemInfoPointer(types.voidptr)),
            # A pointer to the owner python str/unicode object
            ('parent', types.pyobject),
        ]
        models.StructModel.__init__(self, dmm, fe_type, members)


make_attribute_wrapper(types.UnicodeType, 'data', '_data')
make_attribute_wrapper(types.UnicodeType, 'length', '_length')
make_attribute_wrapper(types.UnicodeType, 'kind', '_kind')
make_attribute_wrapper(types.UnicodeType, 'hash', '_hash')


### CAST


def compile_time_get_string_data(obj):
    """Get string data from a python string for use at compile-time to embed
    the string data into the LLVM module.
    """
    from ctypes import (
        CFUNCTYPE, c_void_p, c_int, c_ssize_t, c_ubyte, py_object,
        POINTER, byref,
    )

    extract_unicode_fn = c_helpers['extract_unicode']
    proto = CFUNCTYPE(c_void_p, py_object, POINTER(c_ssize_t), POINTER(c_int),
                      POINTER(c_ssize_t))
    fn = proto(extract_unicode_fn)
    length = c_ssize_t()
    kind = c_int()
    hashv = c_ssize_t()
    data = fn(obj, byref(length), byref(kind), byref(hashv))
    if data is None:
        raise ValueError("cannot extract unicode data from the given string")
    length = length.value
    kind = kind.value
    nbytes = (length + 1) * _kind_to_byte_width(kind)
    out = (c_ubyte * nbytes).from_address(data)
    return bytes(out), length, kind, hashv.value


def make_string_from_constant(context, builder, typ, literal_string):
    """
    Get string data by `compile_time_get_string_data()` and return a
    unicode_type LLVM value
    """
    databytes, length, kind, hashv = \
        compile_time_get_string_data(literal_string)
    mod = builder.module
    gv = context.insert_const_bytes(mod, databytes)
    uni_str = cgutils.create_struct_proxy(typ)(context, builder)
    uni_str.data = gv
    uni_str.length = uni_str.length.type(length)
    uni_str.kind = uni_str.kind.type(kind)
    uni_str.hash = uni_str.hash.type(hashv)
    return uni_str._getvalue()


@lower_cast(types.StringLiteral, types.unicode_type)
def cast_from_literal(context, builder, fromty, toty, val):
    return make_string_from_constant(
        context, builder, toty, fromty.literal_value,
    )


### CONSTANT

@lower_constant(types.unicode_type)
def constant_unicode(context, builder, typ, pyval):
    return make_string_from_constant(context, builder, typ, pyval)


### BOXING


@unbox(types.UnicodeType)
def unbox_unicode_str(typ, obj, c):
    """
    Convert a unicode str object to a native unicode structure.
    """
    ok, data, length, kind, hashv = c.pyapi.string_as_string_size_and_kind(obj)
    uni_str = cgutils.create_struct_proxy(typ)(c.context, c.builder)
    uni_str.data = data
    uni_str.length = length
    uni_str.kind = kind
    uni_str.hash = hashv
    uni_str.meminfo = c.pyapi.nrt_meminfo_new_from_pyobject(
        data,  # the borrowed data pointer
        obj,   # the owner pyobject; the call will incref it.
    )
    uni_str.parent = obj

    is_error = cgutils.is_not_null(c.builder, c.pyapi.err_occurred())
    return NativeValue(uni_str._getvalue(), is_error=is_error)


@box(types.UnicodeType)
def box_unicode_str(typ, val, c):
    """
    Convert a native unicode structure to a unicode string
    """
    uni_str = cgutils.create_struct_proxy(typ)(c.context, c.builder, value=val)
    res = c.pyapi.string_from_kind_and_data(uni_str.kind, uni_str.data, uni_str.length)
    # hash isn't needed now, just compute it so it ends up in the unicodeobject
    # hash cache, cpython doesn't always do this, depends how a string was
    # created it's safe, just burns the cycles required to hash on @box
    c.pyapi.object_hash(res)
    c.context.nrt.decref(c.builder, typ, val)
    return res


#### HELPER FUNCTIONS


def make_deref_codegen(bitsize):
    def codegen(context, builder, signature, args):
        data, idx = args
        ptr = builder.bitcast(data, IntType(bitsize).as_pointer())
        ch = builder.load(builder.gep(ptr, [idx]))
        return builder.zext(ch, IntType(32))

    return codegen


@intrinsic
def deref_uint8(typingctx, data, offset):
    sig = types.uint32(types.voidptr, types.intp)
    return sig, make_deref_codegen(8)


@intrinsic
def deref_uint16(typingctx, data, offset):
    sig = types.uint32(types.voidptr, types.intp)
    return sig, make_deref_codegen(16)


@intrinsic
def deref_uint32(typingctx, data, offset):
    sig = types.uint32(types.voidptr, types.intp)
    return sig, make_deref_codegen(32)


@intrinsic
def _malloc_string(typingctx, kind, char_bytes, length):
    """make empty string with data buffer of size alloc_bytes.

    Must set length and kind values for string after it is returned
    """
    def details(context, builder, signature, args):
        [kind_val, char_bytes_val, length_val] = args

        # fill the struct
        uni_str_ctor = cgutils.create_struct_proxy(types.unicode_type)
        uni_str = uni_str_ctor(context, builder)
        # add null padding character
        nbytes_val = builder.mul(char_bytes_val,
                                 builder.add(length_val,
                                             Constant(length_val.type, 1)))
        uni_str.meminfo = context.nrt.meminfo_alloc(builder, nbytes_val)
        uni_str.kind = kind_val
        uni_str.length = length_val
        # empty string has hash value -1 to indicate "need to compute hash"
        uni_str.hash = context.get_constant(_Py_hash_t, -1)
        uni_str.data = context.nrt.meminfo_data(builder, uni_str.meminfo)
        # Set parent to NULL
        uni_str.parent = cgutils.get_null_value(uni_str.parent.type)
        return uni_str._getvalue()

    sig = types.unicode_type(types.int32, types.intp, types.intp)
    return sig, details


@njit
def _empty_string(kind, length):
    char_width = _kind_to_byte_width(kind)
    s = _malloc_string(kind, char_width, length)
    _set_code_point(s, length, np.uint32(0))    # Write NULL character
    return s


# Disable RefCt for performance.
@njit(_nrt=False)
def _get_code_point(a, i):
    if a._kind == PY_UNICODE_1BYTE_KIND:
        return deref_uint8(a._data, i)
    elif a._kind == PY_UNICODE_2BYTE_KIND:
        return deref_uint16(a._data, i)
    elif a._kind == PY_UNICODE_4BYTE_KIND:
        return deref_uint32(a._data, i)
    else:
        # there's also a wchar kind, but that's one of the above,
        # so skipping for this example
        return 0

####


def make_set_codegen(bitsize):
    def codegen(context, builder, signature, args):
        data, idx, ch = args
        if bitsize < 32:
            ch = builder.trunc(ch, IntType(bitsize))
        ptr = builder.bitcast(data, IntType(bitsize).as_pointer())
        builder.store(ch, builder.gep(ptr, [idx]))
        return context.get_dummy_value()

    return codegen


@intrinsic
def set_uint8(typingctx, data, idx, ch):
    sig = types.void(types.voidptr, types.int64, types.uint32)
    return sig, make_set_codegen(8)


@intrinsic
def set_uint16(typingctx, data, idx, ch):
    sig = types.void(types.voidptr, types.int64, types.uint32)
    return sig, make_set_codegen(16)


@intrinsic
def set_uint32(typingctx, data, idx, ch):
    sig = types.void(types.voidptr, types.int64, types.uint32)
    return sig, make_set_codegen(32)


@njit(_nrt=False)
def _set_code_point(a, i, ch):
    ### WARNING: This method is very dangerous:
    #   * Assumes that data contents can be changed (only allowed for new
    #     strings)
    #   * Assumes that the kind of unicode string is sufficiently wide to
    #     accept ch.  Will truncate ch to make it fit.
    #   * Assumes that i is within the valid boundaries of the function
    if a._kind == PY_UNICODE_1BYTE_KIND:
        set_uint8(a._data, i, ch)
    elif a._kind == PY_UNICODE_2BYTE_KIND:
        set_uint16(a._data, i, ch)
    elif a._kind == PY_UNICODE_4BYTE_KIND:
        set_uint32(a._data, i, ch)
    else:
        raise AssertionError("Unexpected unicode representation in _set_code_point")


@njit
def _pick_kind(kind1, kind2):
    if kind1 == PY_UNICODE_WCHAR_KIND or kind2 == PY_UNICODE_WCHAR_KIND:
        raise AssertionError("PY_UNICODE_WCHAR_KIND unsupported")

    if kind1 == PY_UNICODE_1BYTE_KIND:
        return kind2
    elif kind1 == PY_UNICODE_2BYTE_KIND:
        if kind2 == PY_UNICODE_4BYTE_KIND:
            return kind2
        else:
            return kind1
    elif kind1 == PY_UNICODE_4BYTE_KIND:
        return kind1
    else:
        raise AssertionError("Unexpected unicode representation in _pick_kind")


@njit
def _kind_to_byte_width(kind):
    if kind == PY_UNICODE_1BYTE_KIND:
        return 1
    elif kind == PY_UNICODE_2BYTE_KIND:
        return 2
    elif kind == PY_UNICODE_4BYTE_KIND:
        return 4
    elif kind == PY_UNICODE_WCHAR_KIND:
        raise AssertionError("PY_UNICODE_WCHAR_KIND unsupported")
    else:
        raise AssertionError("Unexpected unicode encoding encountered")


@njit(_nrt=False)
def _cmp_region(a, a_offset, b, b_offset, n):
    if n == 0:
        return 0
    elif a_offset + n > a._length:
        return -1
    elif b_offset + n > b._length:
        return 1

    for i in range(n):
        a_chr = _get_code_point(a, a_offset + i)
        b_chr = _get_code_point(b, b_offset + i)
        if a_chr < b_chr:
            return -1
        elif a_chr > b_chr:
            return 1

    return 0


@njit(_nrt=False)
def _find(substr, s):
    # Naive, slow string matching for now
    for i in range(len(s) - len(substr) + 1):
        if _cmp_region(s, i, substr, 0, len(substr)) == 0:
            return i
    return -1


@njit
def _is_whitespace(code_point):
    # list copied from https://github.com/python/cpython/blob/master/Objects/unicodetype_db.h
    return code_point == 0x0009 \
        or code_point == 0x000A \
        or code_point == 0x000B \
        or code_point == 0x000C \
        or code_point == 0x000D \
        or code_point == 0x001C \
        or code_point == 0x001D \
        or code_point == 0x001E \
        or code_point == 0x001F \
        or code_point == 0x0020 \
        or code_point == 0x0085 \
        or code_point == 0x00A0 \
        or code_point == 0x1680 \
        or code_point == 0x2000 \
        or code_point == 0x2001 \
        or code_point == 0x2002 \
        or code_point == 0x2003 \
        or code_point == 0x2004 \
        or code_point == 0x2005 \
        or code_point == 0x2006 \
        or code_point == 0x2007 \
        or code_point == 0x2008 \
        or code_point == 0x2009 \
        or code_point == 0x200A \
        or code_point == 0x2028 \
        or code_point == 0x2029 \
        or code_point == 0x202F \
        or code_point == 0x205F \
        or code_point == 0x3000


#### PUBLIC API

@overload(len)
def unicode_len(s):
    if isinstance(s, types.UnicodeType):
        def len_impl(s):
            return s._length
        return len_impl


@overload(operator.eq)
def unicode_eq(a, b):
    if isinstance(a, types.UnicodeType) and isinstance(b, types.UnicodeType):
        def eq_impl(a, b):
            if len(a) != len(b):
                return False
            return _cmp_region(a, 0, b, 0, len(a)) == 0
        return eq_impl


@overload(operator.ne)
def unicode_ne(a, b):
    if isinstance(a, types.UnicodeType) and isinstance(b, types.UnicodeType):
        def ne_impl(a, b):
            return not (a == b)
        return ne_impl


@overload(operator.lt)
def unicode_lt(a, b):
    if isinstance(a, types.UnicodeType) and isinstance(b, types.UnicodeType):
        def lt_impl(a, b):
            minlen = min(len(a), len(b))
            eqcode = _cmp_region(a, 0, b, 0, minlen)
            if eqcode == -1:
                return True
            elif eqcode == 0:
                return len(a) < len(b)
            return False
        return lt_impl


@overload(operator.gt)
def unicode_gt(a, b):
    if isinstance(a, types.UnicodeType) and isinstance(b, types.UnicodeType):
        def gt_impl(a, b):
            minlen = min(len(a), len(b))
            eqcode = _cmp_region(a, 0, b, 0, minlen)
            if eqcode == 1:
                return True
            elif eqcode == 0:
                return len(a) > len(b)
            return False
        return gt_impl


@overload(operator.le)
def unicode_le(a, b):
    if isinstance(a, types.UnicodeType) and isinstance(b, types.UnicodeType):
        def le_impl(a, b):
            return not (a > b)
        return le_impl


@overload(operator.ge)
def unicode_ge(a, b):
    if isinstance(a, types.UnicodeType) and isinstance(b, types.UnicodeType):
        def ge_impl(a, b):
            return not (a < b)
        return ge_impl


@overload(operator.contains)
def unicode_contains(a, b):
    if isinstance(a, types.UnicodeType) and isinstance(b, types.UnicodeType):
        def contains_impl(a, b):
            # note parameter swap: contains(a, b) == b in a
            return _find(substr=b, s=a) > -1
        return contains_impl


@overload_method(types.UnicodeType, 'find')
def unicode_find(a, b):
    if isinstance(b, types.UnicodeType):
        def find_impl(a, b):
            return _find(substr=b, s=a)
        return find_impl


@overload_method(types.UnicodeType, 'startswith')
def unicode_startswith(a, b):
    if isinstance(b, types.UnicodeType):
        def startswith_impl(a, b):
            return _cmp_region(a, 0, b, 0, len(b)) == 0
        return startswith_impl


@overload_method(types.UnicodeType, 'endswith')
def unicode_endswith(a, b):
    if isinstance(b, types.UnicodeType):
        def endswith_impl(a, b):
            a_offset = len(a) - len(b)
            if a_offset < 0:
                return False
            return _cmp_region(a, a_offset, b, 0, len(b)) == 0
        return endswith_impl


@overload_method(types.UnicodeType, 'split')
def unicode_split(a, sep=None, maxsplit=-1):
    if not (maxsplit == -1 or
            isinstance(maxsplit, (types.Omitted, types.Integer, types.IntegerLiteral))):
        return None  # fail typing if maxsplit is not an integer

    if isinstance(sep, types.UnicodeType):
        def split_impl(a, sep, maxsplit=-1):
            a_len = len(a)
            sep_len = len(sep)

            if sep_len == 0:
                raise ValueError('empty separator')

            parts = []
            last = 0
            idx = 0
            split_count = 0

            while idx < a_len and (maxsplit == -1 or split_count < maxsplit):
                if _cmp_region(a, idx, sep, 0, sep_len) == 0:
                    parts.append(a[last:idx])
                    idx += sep_len
                    last = idx
                    split_count += 1
                else:
                    idx += 1

            if last <= a_len:
                parts.append(a[last:])

            return parts
        return split_impl
    elif sep is None or isinstance(sep, types.NoneType) or getattr(sep, 'value', False) is None:
        def split_whitespace_impl(a, sep=None, maxsplit=-1):
            a_len = len(a)

            parts = []
            last = 0
            idx = 0
            split_count = 0
            in_whitespace_block = True

            for idx in range(a_len):
                code_point = _get_code_point(a, idx)
                is_whitespace = _is_whitespace(code_point)
                if in_whitespace_block:
                    if is_whitespace:
                        pass  # keep consuming space
                    else:
                        last = idx # this is the start of the next string
                        in_whitespace_block = False
                else:
                    if not is_whitespace:
                        pass # keep searching for whitespace transition
                    else:
                        parts.append(a[last:idx])
                        in_whitespace_block = True
                        split_count += 1
                        if maxsplit != -1 and split_count == maxsplit:
                            break

            if last <= a_len and not in_whitespace_block:
                parts.append(a[last:])

            return parts
        return split_whitespace_impl


@njit
def join_list(sep, parts):
    parts_len = len(parts)
    if parts_len == 0:
        return ''

    # Precompute size and char_width of result
    sep_len = len(sep)
    length = (parts_len - 1) * sep_len
    kind = sep._kind
    for p in parts:
        length += len(p)
        kind = _pick_kind(kind, p._kind)

    result = _empty_string(kind, length)

    # populate string
    part = parts[0]
    _strncpy(result, 0, part, 0, len(part))
    dst_offset = len(part)
    for idx in range(1, parts_len):
        _strncpy(result, dst_offset, sep, 0, sep_len)
        dst_offset += sep_len
        part = parts[idx]
        _strncpy(result, dst_offset, part, 0, len(part))
        dst_offset += len(part)

    return result


@overload_method(types.UnicodeType, 'join')
def unicode_join(sep, parts):
    if isinstance(parts, types.List):
        if isinstance(parts.dtype, types.UnicodeType):
            def join_list_impl(sep, parts):
                return join_list(sep, parts)
            return join_list_impl
        else:
            pass # lists of any other type not supported
    elif isinstance(parts, types.IterableType):
        def join_iter_impl(sep, parts):
            parts_list = [p for p in parts]
            return join_list(sep, parts_list)
        return join_iter_impl
    elif isinstance(parts, types.UnicodeType):
        # Temporary workaround until UnicodeType is iterable
        def join_str_impl(sep, parts):
            parts_list = [parts[i] for i in range(len(parts))]
            return join_list(sep, parts_list)
        return join_str_impl


### String creation

@njit
def normalize_str_idx(idx, length, is_start=True):
    """
    Parameters
    ----------
    idx : int or None
        the index
    length : int
        the string length
    is_start : bool; optional with defaults to True
        Is it the *start* or the *stop* of the slice?

    Returns
    -------
    norm_idx : int
        normalized index
    """
    if idx is None:
        if is_start:
            return 0
        else:
            return length
    elif idx < 0:
        idx += length

    if idx < 0 or idx >= length:
        raise IndexError("string index out of range")

    return idx


@intrinsic
def _normalize_slice(typingctx, sliceobj, length):
    """Fix slice object.
    """
    sig = sliceobj(sliceobj, length)

    def codegen(context, builder, sig, args):
        [slicetype, lengthtype] = sig.args
        [sliceobj, length] = args
        slice = context.make_helper(builder, slicetype, sliceobj)
        slicing.guard_invalid_slice(context, builder, slicetype, slice)
        slicing.fix_slice(builder, slice, length)
        return slice._getvalue()

    return sig, codegen


@intrinsic
def _slice_span(typingctx, sliceobj):
    """Compute the span from the given slice object.
    """
    sig = types.intp(sliceobj)

    def codegen(context, builder, sig, args):
        [slicetype] = sig.args
        [sliceobj] = args
        slice = context.make_helper(builder, slicetype, sliceobj)
        result_size = slicing.get_slice_length(builder, slice)
        return result_size

    return sig, codegen


@njit(_nrt=False)
def _strncpy(dst, dst_offset, src, src_offset, n):
    if src._kind == dst._kind:
        byte_width = _kind_to_byte_width(src._kind)
        src_byte_offset = byte_width * src_offset
        dst_byte_offset = byte_width * dst_offset
        nbytes = n * byte_width
        memcpy_region(dst._data, dst_byte_offset, src._data, src_byte_offset, nbytes, align=1)
    else:
        for i in range(n):
            _set_code_point(dst, dst_offset + i, _get_code_point(src, src_offset + i))


@overload(operator.getitem)
def unicode_getitem(s, idx):
    if isinstance(s, types.UnicodeType):
        if isinstance(idx, types.Integer):
            def getitem_char(s, idx):
                idx = normalize_str_idx(idx, len(s))
                ret = _empty_string(s._kind, 1)
                _set_code_point(ret, 0, _get_code_point(s, idx))
                return ret
            return getitem_char
        elif isinstance(idx, types.SliceType):
            def getitem_slice(s, idx):
                slice_idx = _normalize_slice(idx, len(s))
                span = _slice_span(slice_idx)
                ret = _empty_string(s._kind, span)

                if slice_idx.step == 1:
                    _strncpy(ret, 0, s, slice_idx.start, span)
                else:
                    cur = slice_idx.start
                    for i in range(span):
                        _set_code_point(ret, i, _get_code_point(s, cur))
                        cur += slice_idx.step
                return ret
            return getitem_slice


@overload(operator.add)
@overload(operator.iadd)
def unicode_concat(a, b):
    if isinstance(a, types.UnicodeType) and isinstance(b, types.UnicodeType):
        def concat_impl(a, b):
            new_length = a._length + b._length
            new_kind = _pick_kind(a._kind, b._kind)
            result = _empty_string(new_kind, new_length)
            for i in range(len(a)):
                _set_code_point(result, i, _get_code_point(a, i))
            for j in range(len(b)):
                _set_code_point(result, len(a) + j, _get_code_point(b, j))
            return result
        return concat_impl
