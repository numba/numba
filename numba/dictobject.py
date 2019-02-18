
import ctypes
import operator
from pprint import pprint
import ctypes

from llvmlite import ir

from numba import cgutils
from numba import _helperlib

from numba.extending import (
    overload,
    overload_method,
    intrinsic,
    register_model,
    models,
    make_attribute_wrapper,
    lower_builtin,
)
from numba.targets.imputils import iternext_impl
from numba import types
from numba.types import (
    DictType,
    DictItemsIterableType,
    DictKeysIterableType,
    DictValuesIterableType,
    DictIteratorType,
    Type,
)


ll_dict_type = cgutils.voidptr_t
ll_dictiter_type = cgutils.voidptr_t
ll_status = cgutils.int32_t
ll_ssize_t = cgutils.intp_t
ll_hash = ll_ssize_t
ll_bytes = cgutils.voidptr_t

DKIX_EMPTY = -1


def new_dict():
    raise NotImplementedError


class DictIterState(Type):
    pass


_DictIterState = DictIterState('dict_iterator_state')



@register_model(DictType)
class DictModel(models.StructModel):
    def __init__(self, dmm, fe_type):
        members = [
            ('data', types.voidptr),
        ]
        super(DictModel, self).__init__(dmm, fe_type, members)


@register_model(DictItemsIterableType)
@register_model(DictKeysIterableType)
@register_model(DictValuesIterableType)
@register_model(DictIteratorType)
class DictIterModel(models.StructModel):
    def __init__(self, dmm, fe_type):
        members = [
            ('parent', fe_type.parent),
            ('state', types.voidptr),
        ]
        super(DictIterModel, self).__init__(dmm, fe_type, members)


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
        return context.make_tuple(builder, resty, [ix, out])

    return sig, codegen


@intrinsic
def _dict_popitem(typingctx, d):
    """Wrap numba_dict_popitem
    """

    keyvalty = types.Tuple([d.key_type, d.value_type])
    resty = types.Tuple([types.int32, types.Optional(keyvalty)])
    sig = resty(d)

    def codegen(context, builder, sig, args):
        fnty = ir.FunctionType(
            ll_status,
            [ll_dict_type, ll_bytes, ll_bytes],
        )
        [d] = args
        [td] = sig.args
        fn = ir.Function(builder.module, fnty, name='numba_dict_popitem')

        dm_key = context.data_model_manager[td.key_type]
        dm_val = context.data_model_manager[td.value_type]

        ptr_key = cgutils.alloca_once(builder, dm_key.get_data_type())
        ptr_val = cgutils.alloca_once(builder, dm_val.get_data_type())

        dp = _dict_get_data(context, builder, td, d)
        status = builder.call(
            fn,
            [
                dp,
                _as_bytes(builder, ptr_key),
                _as_bytes(builder, ptr_val),
            ],
        )
        out = context.make_optional_none(builder, keyvalty)
        pout = cgutils.alloca_once_value(builder, out)

        cond = builder.icmp_signed('==', status, status.type(0))
        with builder.if_then(cond):
            key = dm_key.load_from_data_pointer(builder, ptr_key)
            val = dm_val.load_from_data_pointer(builder, ptr_val)
            keyval = context.make_tuple(builder, keyvalty, [key, val])
            optkeyval = context.make_optional_value(builder, keyvalty, keyval)
            builder.store(optkeyval, pout)

        out = builder.load(pout)
        return cgutils.pack_struct(builder, [status, out])

    return sig, codegen

@intrinsic
def _dict_delitem(typingctx, d, hk, ix):
    """Wrap numba_dict_delitem
    """
    resty = types.int32
    sig = resty(d, hk, types.intp)

    def codegen(context, builder, sig, args):
        fnty = ir.FunctionType(
            ll_status,
            [ll_dict_type, ll_hash, ll_ssize_t],
        )
        [d, hk, ix] = args
        [td, thk, tix] = sig.args

        fn = ir.Function(builder.module, fnty, name='numba_dict_delitem')

        dp = _dict_get_data(context, builder, td, d)
        status = builder.call(fn, [dp, hk, ix])
        return status

    return sig, codegen


def _iterator_codegen(resty):

    def codegen(context, builder, sig, args):
        [d] = args
        [td] = sig.args
        iterhelper = context.make_helper(builder, resty)
        iterhelper.parent = d
        iterhelper.state = iterhelper.state.type(None)
        return iterhelper._getvalue()

    return codegen


@intrinsic
def _dict_items(typingctx, d):
    """Get dictionary iterator for .items()"""
    resty = types.DictItemsIterableType(d)
    sig = resty(d)
    codegen = _iterator_codegen(resty)
    return sig, codegen


@intrinsic
def _dict_keys(typingctx, d):
    """Get dictionary iterator for .keys()"""
    resty = types.DictKeysIterableType(d)
    sig = resty(d)
    codegen = _iterator_codegen(resty)
    return sig, codegen


@intrinsic
def _dict_values(typingctx, d):
    """Get dictionary iterator for .values()"""
    resty = types.DictValuesIterableType(d)
    sig = resty(d)
    codegen = _iterator_codegen(resty)
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


@intrinsic
def _nonoptional(typingctx, val):
    if not isinstance(val, types.Optional):
        raise TypeError('expected an optional')

    def codegen(context, builder, sig, args):
        return args[0]

    casted = val.type
    sig = casted(casted)
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
def impl_get(dct, k, d=None):
    if not isinstance(dct, types.DictType):
        return
    keyty = dct.key_type

    def impl(dct, k, d=None):
        k = _cast(k, keyty)
        ix, val = _dict_lookup(dct, k, hash(k))
        if ix > DKIX_EMPTY:
            return val
        return d

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


@overload_method(types.DictType, 'popitem')
def impl_popitem(d):
    if not isinstance(d, types.DictType):
        return

    def impl(d):
        status, keyval = _dict_popitem(d)
        if status == 0:
            return _nonoptional(keyval)
        elif status == -4:
            raise KeyError()
        else:
            raise AssertionError('internal dict error during popitem')

    return impl


@overload_method(types.DictType, 'pop')
def impl_pop(dct, k, d=None):
    if not isinstance(dct, types.DictType):
        return

    keyty = dct.key_type
    should_raise = isinstance(d, types.Omitted)

    def impl(dct, k, d=None):
        key = _cast(k, keyty)
        hashed = hash(key)
        ix, val = _dict_lookup(dct, key, hashed)
        if ix == DKIX_EMPTY:
            if should_raise:
                raise KeyError()
            else:
                return d
        elif ix < DKIX_EMPTY:
            raise AssertionError("internal dict error during lookup")
        else:
            status = _dict_delitem(dct,hashed, ix)
            if status != 0:
                raise AssertionError("internal dict error during delitem")
            return val

    return impl


# @overload(operator.delitem)
# def impl_delitem(d, key):
#     if not isinstance(d, types.DictType):
#         return
#     print(">>>>>>>")



@overload_method(types.DictType, 'clear')
def impl_clear(d):
    if not isinstance(d, types.DictType):
        return

    def impl(d):
        while len(d):
            d.popitem()

    return impl


@overload_method(types.DictType, 'items')
def impl_items(d):
    if not isinstance(d, types.DictType):
        return

    def impl(d):
        it = _dict_items(d)
        return it

    return impl


@overload_method(types.DictType, 'keys')
def impl_keys(d):
    if not isinstance(d, types.DictType):
        return

    def impl(d):
        return _dict_keys(d)

    return impl


@overload_method(types.DictType, 'values')
def impl_values(d):
    if not isinstance(d, types.DictType):
        return

    def impl(d):
        return _dict_values(d)

    return impl


@lower_builtin('getiter', types.DictItemsIterableType)
@lower_builtin('getiter', types.DictKeysIterableType)
@lower_builtin('getiter', types.DictValuesIterableType)
def impl_items_getiter(context, builder, sig, args):
    itemiterablety = sig.args[0]
    it = context.make_helper(builder, itemiterablety.iterator_type, args[0])

    fnty = ir.FunctionType(
        ir.VoidType(),
        [ll_dictiter_type, ll_dict_type],
    )

    fn = ir.Function(builder.module, fnty, name='numba_dict_iter')

    proto = ctypes.CFUNCTYPE(ctypes.c_size_t)
    dictiter_sizeof = proto(_helperlib.c_helpers['dict_iter_sizeof'])
    state_type = ir.ArrayType(ir.IntType(8), dictiter_sizeof())

    pstate = cgutils.alloca_once(builder, state_type, zfill=True)
    it.state = _as_bytes(builder, pstate)

    dp = _dict_get_data(context, builder, itemiterablety.parent, it.parent)
    builder.call(fn, [it.state, dp])
    return it._getvalue()


@lower_builtin('iternext', types.DictIteratorType)
@iternext_impl
def impl_items_iternext(context, builder, sig, args, result):
    iter_type = sig.args[0]
    it = context.make_helper(builder, iter_type, args[0])

    p2p_bytes = ll_bytes.as_pointer()

    iternext_fnty = ir.FunctionType(
        ll_status,
        [ll_bytes, p2p_bytes, p2p_bytes]
    )
    iternext = ir.Function(
        builder.module,
        iternext_fnty,
        name='numba_dict_iter_next',
    )
    key_raw_ptr = cgutils.alloca_once(builder, ll_bytes)
    val_raw_ptr = cgutils.alloca_once(builder, ll_bytes)

    status = builder.call(iternext, (it.state, key_raw_ptr, val_raw_ptr))
    # TODO: no handling of error state i.e. mutated dictionary
    #       any error are treated as exhausted iterator
    is_valid = builder.icmp_unsigned('==', status, status.type(0))
    result.set_valid(is_valid)

    with builder.if_then(is_valid):
        yield_type = iter_type.yield_type
        key_ty, val_ty = iter_type.parent.keyvalue_type

        dm_key = context.data_model_manager[key_ty]
        dm_val = context.data_model_manager[val_ty]

        key_ptr = builder.bitcast(
            builder.load(key_raw_ptr),
            dm_key.get_data_type().as_pointer(),
        )
        val_ptr = builder.bitcast(
            builder.load(val_raw_ptr),
            dm_val.get_data_type().as_pointer(),
        )

        key = dm_key.load_from_data_pointer(builder, key_ptr)
        val = dm_val.load_from_data_pointer(builder, val_ptr)

        if isinstance(iter_type.iterable, DictItemsIterableType):
            tup = context.make_tuple(builder, yield_type, [key, val])
            result.yield_(tup)
        elif isinstance(iter_type.iterable, DictKeysIterableType):
            result.yield_(key)
        elif isinstance(iter_type.iterable, DictValuesIterableType):
            result.yield_(val)
        else:
            raise AssertionError('unknown type: {}'.format(iter_type.iterable))
