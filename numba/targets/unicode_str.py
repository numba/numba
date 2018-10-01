from numba.extending import (models, register_model,
    make_attribute_wrapper, unbox, box, NativeValue, overload,
    lower_getattr, overload_method, intrinsic)
from numba import cgutils
from numba import types
from numba import njit
from numba.pythonapi import (PY_UNICODE_1BYTE_KIND, PY_UNICODE_2BYTE_KIND,
    PY_UNICODE_4BYTE_KIND, PY_UNICODE_WCHAR_KIND)
from llvmlite.ir import IntType
import operator


### DATA MODEL

@register_model(types.UnicodeType)
class UnicodeModel(models.StructModel):
    def __init__(self, dmm, fe_type):
        members = [
            ('data', types.voidptr),
            ('length', types.int64),
            ('kind', types.int64),
            ('meminfo', types.MemInfoPointer(types.voidptr)),
            # A pointer to the owner python str/unicode object
            ('parent', types.pyobject),
            ]
        models.StructModel.__init__(self, dmm, fe_type, members)


make_attribute_wrapper(types.UnicodeType, 'data', '_data')
make_attribute_wrapper(types.UnicodeType, 'length', '_length')
make_attribute_wrapper(types.UnicodeType, 'kind', '_kind')


### BOXING


@unbox(types.UnicodeType)
def unbox_unicode_str(typ, obj, c):
    """
    Convert a unicode str object to a native unicode structure.
    """
    ok, data, length, kind  = c.pyapi.string_as_string_size_and_kind(obj)
    uni_str = cgutils.create_struct_proxy(typ)(c.context, c.builder)
    uni_str.data = data
    uni_str.length = length
    uni_str.kind = kind
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
    sig = types.uint32(types.voidptr, types.int64)
    return sig, make_deref_codegen(8)


@intrinsic
def deref_uint16(typingctx, data, offset):
    sig = types.uint32(types.voidptr, types.int64)
    return sig, make_deref_codegen(16)


@intrinsic
def deref_uint32(typingctx, data, offset):
    sig = types.uint32(types.voidptr, types.int64)
    return sig, make_deref_codegen(32)


@njit
def _get_code_point(a, i):
    if a._kind == PY_UNICODE_1BYTE_KIND:
        return deref_uint8(a._data, i)
    elif a._kind == PY_UNICODE_2BYTE_KIND:
        return deref_uint16(a._data, i)
    elif a._kind == PY_UNICODE_4BYTE_KIND:
        return deref_uint32(a._data, i)
    else:
        return 0 # there's also a wchar kind, but that's one of the above, so skipping for this example


@njit
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


@njit
def _cmp(a, b):
    if len(a) < len(b):
        return -1
    elif len(a) > len(b):
        return 1
    else:
        return _cmp_region(a, b)


@njit
def _find(substr, s):
    # Naive, slow string matching for now
    for i in range(len(s) - len(substr) + 1):
        if _cmp_region(s, i, substr, 0, len(substr)) == 0:
            return i
    return -1


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


@overload(operator.lt)
def unicode_lt(a, b):
    if isinstance(a, types.UnicodeType) and isinstance(b, types.UnicodeType):
        def lt_impl(a, b):
            if len(a) > len(b):
                return False
            elif len(a) < len(b):
                return True
            else:
                return _cmp_region(a, 0, b, 0, len(a)) == -1
        return lt_impl


@overload(operator.gt)
def unicode_gt(a, b):
    if isinstance(a, types.UnicodeType) and isinstance(b, types.UnicodeType):
        def gt_impl(a, b):
            if len(a) > len(b):
                return True
            elif len(a) < len(b):
                return False
            return _cmp_region(a, 0, b, 0, len(a)) == 1
        return gt_impl


@overload(operator.le)
def unicode_le(a, b):
    if isinstance(a, types.UnicodeType) and isinstance(b, types.UnicodeType):
        def le_impl(a, b):
            if len(a) > len(b):
                return False
            elif len(a) < len(b):
                return True
            ret = _cmp_region(a, 0, b, 0, len(a))
            return (ret == -1) or (ret == 0)
        return le_impl


@overload(operator.ge)
def unicode_ge(a, b):
    if isinstance(a, types.UnicodeType) and isinstance(b, types.UnicodeType):
        def ge_impl(a, b):
            if len(a) > len(b):
                return True
            elif len(a) < len(b):
                return False
            ret = _cmp_region(a, 0, b, 0, len(a))
            return (ret == 1) or (ret == 0)
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


