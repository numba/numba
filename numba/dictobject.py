
import ctypes
import operator
from pprint import pprint

from llvmlite import ir

from numba import cgutils
from numba.extending import (
    overload,
    overload_method,
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
ll_hash = ll_ssize_t
ll_bytes = cgutils.voidptr_t

DKIX_EMPTY = -1


def new_dict():
    raise NotImplementedError


@register_model(DictType)
class DictModel(models.StructModel):
    def __init__(self, dmm, fe_type):
        members = [
            ('data', types.voidptr),
        ]
        super(DictModel, self).__init__(dmm, fe_type, members)


make_attribute_wrapper(types.DictType, 'data', '_data')


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
        ll_key = context.get_data_type(keyty.instance_type)
        ll_val = context.get_data_type(valty.instance_type)
        sz_key = context.get_abi_sizeof(ll_key)
        sz_val = context.get_abi_sizeof(ll_val)
        refdp = cgutils.alloca_once(builder, ll_dict_type, zfill=True)
        status = builder.call(fn, [refdp, ll_ssize_t(sz_key), ll_ssize_t(sz_val)])
        dp = builder.load(refdp)
        return dp

    return sig, codegen


@intrinsic
def _dict_insert(typingctx, d, key, hashval, val):
    resty = types.int32
    sig = resty(d, d.key_type, types.intp, d.value_type)


    def codegen(context, builder, sig, args):
        fnty = ir.FunctionType(
            ll_status,
            [ll_dict_type, ll_bytes, ll_hash, ll_bytes, ll_bytes],
        )
        [d, key, hashval, val] = args
        [td, tkey, thashval, tval] = sig.args
        fn = ir.Function(builder.module, fnty, name='numba_dict_insert')

        dm_key = context.data_model_manager[tkey]
        dm_val = context.data_model_manager[tval]

        data_key = dm_key.as_data(builder, key)
        data_val = dm_val.as_data(builder, val)

        ptr_key = cgutils.alloca_once_value(builder, data_key)
        ptr_val = cgutils.alloca_once_value(builder, data_val)
        ptr_oldval = cgutils.alloca_once(builder, data_val.type)

        dp = _dict_get_data(context, builder, td, d)
        status = builder.call(
            fn,
            [
                dp,
                _as_bytes(builder, ptr_key),
                hashval,
                _as_bytes(builder, ptr_val),
                _as_bytes(builder, ptr_oldval),
            ],
        )
        return status

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
def _dict_dump_keys(typingctx, d):
    """Dump the dictionary key and values.
    Wraps numba_dict_dump_keys for debugging.
    """
    resty = types.void
    sig = resty(d)

    def codegen(context, builder, sig, args):
        fnty = ir.FunctionType(
            ir.VoidType(),
            [ll_dict_type],
        )
        [td] = sig.args
        [d] = args
        dp = _dict_get_data(context, builder, td, d)
        fn = ir.Function(builder.module, fnty, name='numba_dict_dump_keys')

        builder.call(fn, [dp])

    return sig, codegen

@intrinsic
def _dict_lookup(typingctx, d, key, hashval):
    """Wrap numba_dict_lookup

    Returns 2-tuple of (intp, value_type?)
    """
    resty = types.Tuple([types.intp, types.Optional(d.value_type)])
    sig = resty(d, key, hashval)

    def codegen(context, builder, sig, args):
        fnty = ir.FunctionType(
            ll_ssize_t,
            [ll_dict_type, ll_bytes, ll_hash, ll_bytes],
        )
        [td, tkey, thashval] = sig.args
        [d, key, hashval] = args
        fn = ir.Function(builder.module, fnty, name='numba_dict_lookup')

        dm_key = context.data_model_manager[tkey]
        dm_val = context.data_model_manager[td.value_type]

        data_key = dm_key.as_data(builder, key)
        ptr_key = cgutils.alloca_once_value(builder, data_key)

        ll_val = context.get_data_type(td.value_type)
        ptr_val = cgutils.alloca_once(builder, ll_val)

        dp = _dict_get_data(context, builder, td, d)
        ix = builder.call(
            fn,
            [
                dp,
                _as_bytes(builder, ptr_key),
                hashval,
                _as_bytes(builder, ptr_val),
            ],
        )
        # Load value is output is available
        found = builder.icmp_signed('>=', ix, ix.type(DKIX_EMPTY))

        out = context.make_optional_none(builder, td.value_type)
        pout = cgutils.alloca_once_value(builder, out)

        with builder.if_then(found):
            val = dm_val.load_from_data_pointer(builder, ptr_val)
            loaded = context.make_optional_value(builder, td.value_type, val)
            builder.store(loaded, pout)

        out = builder.load(pout)
        return cgutils.pack_struct(builder, [ix, out])

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
    dict_ty = types.DictType(keyty.instance_type, valty.instance_type)

    def codegen(context, builder, signature, args):
        [_, _, ptr] = args
        ctor = cgutils.create_struct_proxy(dict_ty)
        dstruct = ctor(context, builder)
        dstruct.data = ptr
        return dstruct._getvalue()

    sig = dict_ty(keyty, valty, ptr)
    return sig, codegen


def _dict_get_data(context, builder, dict_ty, d):
    ctor = cgutils.create_struct_proxy(dict_ty)
    dstruct = ctor(context, builder, value=d)
    return dstruct.data


def _as_bytes(builder, ptr):
    return builder.bitcast(ptr, cgutils.voidptr_t)


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


@intrinsic
def _cast(typingctx, val, typ):
    def codegen(context, builder, signature, args):
        [val, typ] = args
        return val
    casted = typ.instance_type
    sig = casted(casted, typ)
    return sig, codegen



@overload(operator.setitem)
def impl_setitem(d, key, value):
    if not isinstance(d, types.DictType):
        return

    keyty, valty = d.key_type, d.value_type

    def impl(d, key, value):
        key = _cast(key, keyty)
        val = _cast(value, valty)
        status = _dict_insert(d, key, hash(key), val)

    return impl


@overload_method(types.DictType, 'get')
def impl_get(d, key):
    if not isinstance(d, types.DictType):
        return
    keyty = d.key_type

    def impl(d, key):
        key = _cast(key, keyty)
        ix, val = _dict_lookup(d, key, hash(key))
        if ix > DKIX_EMPTY:
            return val

    return impl


@overload(operator.getitem)
def impl_getitem(d, key):
    if not isinstance(d, types.DictType):
        return

    keyty = d.key_type

    def impl(d, key):
        key = _cast(key, keyty)
        ix, val = _dict_lookup(d, key, hash(key))
        if ix == DKIX_EMPTY:
            raise KeyError()
        elif ix < DKIX_EMPTY:
            raise AssertionError("internal dict error during lookup")
        else:
            return val

    return impl


