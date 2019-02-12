
import ctypes
import operator
from pprint import pprint

from llvmlite import ir

from numba import cgutils
from numba.extending import (
    overload,
    intrinsic,
    register_model,
    models,
    make_attribute_wrapper,
)
from numba import types
from numba.types import DictType, Type


ll_dict_type = cgutils.voidptr_t
ll_status = cgutils.int32_t
ll_ssize_t = cgutils.intp_t



@register_model(DictType)
class DictModel(models.StructModel):
    def __init__(self, dmm, fe_type):
        members = [
            ('data', types.voidptr),
        ]
        super(DictModel, self).__init__(dmm, fe_type, members)


make_attribute_wrapper(types.DictType, 'data', '_data')


def new_dict(key_type, value_type):
    raise NotImplementedError


@intrinsic
def _dict_new_minsize(typingctx, keyty, valty):
    """Wrap numba_dict_new_minsize.

    Allocate a new dictionary object at the minimum capacity.

    Parameters
    ----------
    keyty, valty: Type
        Type of the key and value, respectively.

    """
    resty = types.voidptr
    sig = resty(keyty, valty)

    def codegen(context, builder, sig, args):
        fnty = ir.FunctionType(
            ll_status,
            [ll_dict_type.as_pointer(), ll_ssize_t, ll_ssize_t],
        )
        fn = ir.Function(builder.module, fnty, name='numba_dict_new_minsize')
        ll_key = context.get_data_type(keyty)
        ll_val = context.get_data_type(valty)
        sz_key = context.get_abi_sizeof(ll_key)
        sz_val = context.get_abi_sizeof(ll_val)
        refdp = cgutils.alloca_once(builder, ll_dict_type, zfill=True)
        status = builder.call(fn, [refdp, ll_ssize_t(sz_key), ll_ssize_t(sz_val)])
        dp = builder.load(refdp)
        return dp

    return sig, codegen


@intrinsic
def _dict_length(typingctx, d):
    """Wrap numba_dict_length

    Returns the length of the dictionary.
    """
    resty = types.intp
    sig = resty(d)

    def codegen(context, builder, sig, args):
        fnty = ir.FunctionType(
            ll_ssize_t,
            [ll_dict_type],
        )
        fn = ir.Function(builder.module, fnty, name='numba_dict_length')
        n = builder.call(fn, args)
        return n

    return sig, codegen


@intrinsic
def _make_dict(typingctx, keyty, valty, ptr):
    """Make a dictionary struct with the given *ptr*

    Parameters
    ----------
    keyty, valty: Type
        Type of the key and value, respectively.
    ptr : llvm pointer value
        Points to the dictionary object.
    """
    dict_ty = types.DictType(keyty.dtype, valty.dtype)

    def codegen(context, builder, signature, args):
        [_, _, ptr] = args
        ctor = cgutils.create_struct_proxy(dict_ty)
        dstruct = ctor(context, builder)
        dstruct.data = ptr
        return dstruct._getvalue()

    sig = dict_ty(keyty, valty, ptr)
    return sig, codegen


@overload(new_dict)
def impl_new_dict(key, value):
    """Creates a new dictionary with *key* and *value* as the type
    of the dictionary key and value, respectively.
    """
    if any([
        not isinstance(key, Type),
        not isinstance(value, Type),
    ]):
        raise TypeError

    keyty, valty = key, value

    def imp(key, value):
        dp = _dict_new_minsize(keyty, valty)
        d = _make_dict(keyty, valty, dp)
        return d

    return imp


@overload(len)
def impl_len(d):
    """len(dict)
    """
    if not isinstance(d, types.DictType):
        return

    def impl(d):
        return _dict_length(d._data)

    return impl


@overload(operator.setitem)
def impl_setitem(d, key, value):
    print('>>>', d, key, value)
    if not isinstance(d, types.DictType):
        return

    def impl(d, key, value):
        hashval = hash(key)
        print(hashval)
        # return _dict_insert(d, key, hashval, value)

    return impl
