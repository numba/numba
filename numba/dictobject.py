"""
Compiler-side implementation of the dictionary.
"""
import ctypes
import operator
from enum import IntEnum

from llvmlite import ir

from numba import cgutils
from numba import _helperlib
from numba.targets.registry import cpu_target

from numba.extending import (
    overload,
    overload_method,
    intrinsic,
    register_model,
    models,
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
from numba.typeconv import Conversion
from numba.targets.imputils import impl_ret_borrowed
from numba.errors import TypingError
from numba import typing


ll_dict_type = cgutils.voidptr_t
ll_dictiter_type = cgutils.voidptr_t
ll_voidptr_type = cgutils.voidptr_t
ll_status = cgutils.int32_t
ll_ssize_t = cgutils.intp_t
ll_hash = ll_ssize_t
ll_bytes = cgutils.voidptr_t


_meminfo_dictptr = types.MemInfoPointer(types.voidptr)


# The following enums must match _dictobject.c

class DKIX(IntEnum):
    """Special return value of dict lookup.
    """
    EMPTY = -1


class Status(IntEnum):
    """Status code for other dict operations.
    """
    OK = 0
    OK_REPLACED = 1
    ERR_NO_MEMORY = -1
    ERR_DICT_MUTATED = -2
    ERR_ITER_EXHAUSTED = -3
    ERR_DICT_EMPTY = -4
    ERR_CMP_FAILED = -5


def new_dict(key, value):
    """Construct a new dict. (Not implemented in the interpreter yet)

    Parameters
    ----------
    key, value : TypeRef
        Key type and value type of the new dict.
    """
    raise NotImplementedError


@register_model(DictType)
class DictModel(models.StructModel):
    def __init__(self, dmm, fe_type):
        members = [
            ('meminfo', _meminfo_dictptr),
            ('data', types.voidptr),   # ptr to the C dict
        ]
        super(DictModel, self).__init__(dmm, fe_type, members)


@register_model(DictItemsIterableType)
@register_model(DictKeysIterableType)
@register_model(DictValuesIterableType)
@register_model(DictIteratorType)
class DictIterModel(models.StructModel):
    def __init__(self, dmm, fe_type):
        members = [
            ('parent', fe_type.parent),  # reference to the dict
            ('state', types.voidptr),    # iterator state in C code
        ]
        super(DictIterModel, self).__init__(dmm, fe_type, members)


def _raise_if_error(context, builder, status, msg):
    """Raise an internal error depending on the value of *status*
    """
    ok_status = status.type(int(Status.OK))
    with builder.if_then(builder.icmp_signed('!=', status, ok_status)):
        context.call_conv.return_user_exc(builder, RuntimeError, (msg,))


@intrinsic
def _as_meminfo(typingctx, dctobj):
    """Returns the MemInfoPointer of a dictionary.
    """
    if not isinstance(dctobj, types.DictType):
        raise TypingError('expected *dctobj* to be a DictType')

    def codegen(context, builder, sig, args):
        [td] = sig.args
        [d] = args
        # Incref
        context.nrt.incref(builder, td, d)
        ctor = cgutils.create_struct_proxy(td)
        dstruct = ctor(context, builder, value=d)
        # Returns the plain MemInfo
        return dstruct.meminfo

    sig = _meminfo_dictptr(dctobj)
    return sig, codegen


@intrinsic
def _from_meminfo(typingctx, mi, dicttyperef):
    """Recreate a dictionary from a MemInfoPointer
    """
    if mi != _meminfo_dictptr:
        raise TypingError('expected a MemInfoPointer for dict.')
    dicttype = dicttyperef.instance_type
    if not isinstance(dicttype, DictType):
        raise TypingError('expected a {}'.format(DictType))

    def codegen(context, builder, sig, args):
        [tmi, tdref] = sig.args
        td = tdref.instance_type
        [mi, _] = args

        ctor = cgutils.create_struct_proxy(td)
        dstruct = ctor(context, builder)

        data_pointer = context.nrt.meminfo_data(builder, mi)
        data_pointer = builder.bitcast(data_pointer, ll_dict_type.as_pointer())

        dstruct.data = builder.load(data_pointer)
        dstruct.meminfo = mi

        return impl_ret_borrowed(
            context,
            builder,
            dicttype,
            dstruct._getvalue(),
        )

    sig = dicttype(mi, dicttyperef)
    return sig, codegen


def _call_dict_free(context, builder, ptr):
    """Call numba_dict_free(ptr)
    """
    fnty = ir.FunctionType(
        ir.VoidType(),
        [ll_dict_type],
    )
    free = builder.module.get_or_insert_function(fnty, name='numba_dict_free')
    builder.call(free, [ptr])


def _imp_dtor(context, module):
    """Define the dtor for dictionary
    """
    llvoidptr = context.get_value_type(types.voidptr)
    llsize = context.get_value_type(types.uintp)
    fnty = ir.FunctionType(
        ir.VoidType(),
        [llvoidptr, llsize, llvoidptr],
    )
    fname = '_numba_dict_dtor'
    fn = module.get_or_insert_function(fnty, name=fname)

    if fn.is_declaration:
        # Set linkage
        fn.linkage = 'linkonce_odr'
        # Define
        builder = ir.IRBuilder(fn.append_basic_block())
        dp = builder.bitcast(fn.args[0], ll_dict_type.as_pointer())
        d = builder.load(dp)
        _call_dict_free(context, builder, d)
        builder.ret_void()

    return fn


def _dict_get_data(context, builder, dict_ty, d):
    """Helper to get the C dict pointer in a numba dict.
    """
    ctor = cgutils.create_struct_proxy(dict_ty)
    dstruct = ctor(context, builder, value=d)
    return dstruct.data


def _as_bytes(builder, ptr):
    """Helper to do (void*)ptr
    """
    return builder.bitcast(ptr, cgutils.voidptr_t)


def _sentry_safe_cast(fromty, toty):
    """Check and raise TypingError if *fromty* cannot be safely cast to *toty*
    """
    tyctxt = cpu_target.typing_context
    by = tyctxt.can_convert(fromty, toty)
    if by is None or by > Conversion.safe:
        if isinstance(fromty, types.Integer) and isinstance(toty, types.Integer):
            # Accept if both types are ints
            return
        if isinstance(fromty, types.Integer) and isinstance(toty, types.Float):
            # Accept if ints to floats
            return
        if isinstance(fromty, types.Float) and isinstance(toty, types.Float):
            # Accept if floats to floats
            return
        raise TypingError('cannot safely cast {} to {}'.format(fromty, toty))


def _sentry_safe_cast_default(default, valty):
    """Similar to _sentry_safe_cast but handle default value.
    """
    # Handle default values
    # TODO: simplify default values; too many possible way to spell None
    if default is None:
        return
    if isinstance(default, (types.Omitted, types.NoneType)):
        return
    return _sentry_safe_cast(default, valty)


@intrinsic
def _cast(typingctx, val, typ):
    """Cast *val* to *typ*
    """
    def codegen(context, builder, signature, args):
        [val, typ] = args
        context.nrt.incref(builder, signature.return_type, val)
        return val
    # Using implicit casting in argument types
    casted = typ.instance_type
    _sentry_safe_cast(val, casted)
    sig = casted(casted, typ)
    return sig, codegen


@intrinsic
def _nonoptional(typingctx, val):
    """Typing trick to cast Optional[T] to T
    """
    if not isinstance(val, types.Optional):
        raise TypeError('expected an optional')

    def codegen(context, builder, sig, args):
        context.nrt.incref(builder, sig.return_type, args[0])
        return args[0]

    casted = val.type
    sig = casted(casted)
    return sig, codegen


@intrinsic
def _dict_new_minsize(typingctx, keyty, valty):
    """Wrap numba_dict_new_minsize.

    Allocate a new dictionary object with the minimum capacity.

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
        fn = builder.module.get_or_insert_function(fnty, name='numba_dict_new_minsize')
        # Determine sizeof key and value types
        ll_key = context.get_data_type(keyty.instance_type)
        ll_val = context.get_data_type(valty.instance_type)
        sz_key = context.get_abi_sizeof(ll_key)
        sz_val = context.get_abi_sizeof(ll_val)
        refdp = cgutils.alloca_once(builder, ll_dict_type, zfill=True)
        status = builder.call(
            fn,
            [refdp, ll_ssize_t(sz_key), ll_ssize_t(sz_val)],
        )
        _raise_if_error(
            context, builder, status,
            msg="Failed to allocate dictionary",
        )
        dp = builder.load(refdp)
        return dp

    return sig, codegen


def _get_incref_decref(context, module, datamodel):
    assert datamodel.contains_nrt_meminfo()

    fe_type = datamodel.fe_type
    data_ptr_ty = datamodel.get_data_type().as_pointer()
    refct_fnty = ir.FunctionType(ir.VoidType(), [data_ptr_ty])
    incref_fn = module.get_or_insert_function(
        refct_fnty,
        name='.numba_dict_incref${}'.format(fe_type),
    )
    builder = ir.IRBuilder(incref_fn.append_basic_block())
    context.nrt.incref(builder, fe_type, builder.load(incref_fn.args[0]))
    builder.ret_void()

    decref_fn = module.get_or_insert_function(
        refct_fnty,
        name='.numba_dict_decref${}'.format(fe_type),
    )
    builder = ir.IRBuilder(decref_fn.append_basic_block())
    context.nrt.decref(builder, fe_type, builder.load(decref_fn.args[0]))
    builder.ret_void()

    return incref_fn, decref_fn


def _get_equal(context, module, datamodel):
    assert datamodel.contains_nrt_meminfo()

    fe_type = datamodel.fe_type
    data_ptr_ty = datamodel.get_data_type().as_pointer()

    wrapfnty = context.call_conv.get_function_type(types.int32, [fe_type, fe_type])
    argtypes = [fe_type, fe_type]

    def build_wrapper(fn):
        builder = ir.IRBuilder(fn.append_basic_block())
        args = context.call_conv.decode_arguments(builder, argtypes, fn)

        sig = typing.signature(types.boolean, fe_type, fe_type)
        op = operator.eq
        fnop = context.typing_context.resolve_value_type(op)
        fnop.get_call_type(context.typing_context, sig.args, {})
        eqfn = context.get_function(fnop, sig)
        res = eqfn(builder, args)
        intres = context.cast(builder, res, types.boolean, types.int32)
        context.call_conv.return_value(builder, intres)

    wrapfn = module.get_or_insert_function(
        wrapfnty,
        name='.numba_dict_key_equal.wrap${}'.format(fe_type)
    )
    build_wrapper(wrapfn)

    equal_fnty = ir.FunctionType(ir.IntType(32), [data_ptr_ty, data_ptr_ty])
    equal_fn = module.get_or_insert_function(
        equal_fnty,
        name='.numba_dict_key_equal${}'.format(fe_type),
    )
    builder = ir.IRBuilder(equal_fn.append_basic_block())
    lhs = datamodel.load_from_data_pointer(builder, equal_fn.args[0])
    rhs = datamodel.load_from_data_pointer(builder, equal_fn.args[1])

    status, retval = context.call_conv.call_function(
        builder, wrapfn, types.boolean, argtypes, [lhs, rhs],
    )
    with builder.if_then(status.is_ok, likely=True):
        with builder.if_then(status.is_none):
            builder.ret(context.get_constant(types.int32, 0))
        retval = context.cast(builder, retval, types.boolean, types.int32)
        builder.ret(retval)
    # Error out
    builder.ret(context.get_constant(types.int32, -1))

    return equal_fn


@intrinsic
def _dict_set_method_table(typingctx, dp, keyty, valty):
    """Wrap numba_dict_set_method_table
    """
    resty = types.void
    sig = resty(dp, keyty, valty)

    def codegen(context, builder, sig, args):
        vtablety = ir.LiteralStructType([
            ll_voidptr_type,  # equal
            ll_voidptr_type,  # key incref
            ll_voidptr_type,  # key decref
            ll_voidptr_type,  # val incref
            ll_voidptr_type,  # val decref
        ])
        setmethod_fnty = ir.FunctionType(
            ir.VoidType(),
            [ll_dict_type, vtablety.as_pointer()]
        )
        setmethod_fn = ir.Function(
            builder.module,
            setmethod_fnty,
            name='numba_dict_set_method_table',
        )
        dp = args[0]
        vtable = cgutils.alloca_once(builder, vtablety, zfill=True)

        # install key incref/decref
        key_equal_ptr = cgutils.gep_inbounds(builder, vtable, 0, 0)
        key_incref_ptr = cgutils.gep_inbounds(builder, vtable, 0, 1)
        key_decref_ptr = cgutils.gep_inbounds(builder, vtable, 0, 2)
        val_incref_ptr = cgutils.gep_inbounds(builder, vtable, 0, 3)
        val_decref_ptr = cgutils.gep_inbounds(builder, vtable, 0, 4)

        dm_key = context.data_model_manager[keyty.instance_type]
        if dm_key.contains_nrt_meminfo():
            equal = _get_equal(context, builder.module, dm_key)
            key_incref, key_decref = _get_incref_decref(
                context, builder.module, dm_key,
            )
            builder.store(
                builder.bitcast(equal, key_equal_ptr.type.pointee),
                key_equal_ptr,
            )
            builder.store(
                builder.bitcast(key_incref, key_incref_ptr.type.pointee),
                key_incref_ptr,
            )
            builder.store(
                builder.bitcast(key_decref, key_decref_ptr.type.pointee),
                key_decref_ptr,
            )

        dm_val = context.data_model_manager[valty.instance_type]
        if dm_val.contains_nrt_meminfo():
            val_incref, val_decref = _get_incref_decref(
                context, builder.module, dm_val,
            )
            builder.store(
                builder.bitcast(val_incref, val_incref_ptr.type.pointee),
                val_incref_ptr,
            )
            builder.store(
                builder.bitcast(val_decref, val_decref_ptr.type.pointee),
                val_decref_ptr,
            )

        builder.call(setmethod_fn, [dp, vtable])

    return sig, codegen


@intrinsic
def _dict_insert(typingctx, d, key, hashval, val):
    """Wrap numba_dict_insert
    """
    resty = types.int32
    sig = resty(d, d.key_type, types.intp, d.value_type)

    def codegen(context, builder, sig, args):
        fnty = ir.FunctionType(
            ll_status,
            [ll_dict_type, ll_bytes, ll_hash, ll_bytes, ll_bytes],
        )
        [d, key, hashval, val] = args
        [td, tkey, thashval, tval] = sig.args
        fn = builder.module.get_or_insert_function(fnty, name='numba_dict_insert')

        dm_key = context.data_model_manager[tkey]
        dm_val = context.data_model_manager[tval]

        data_key = dm_key.as_data(builder, key)
        data_val = dm_val.as_data(builder, val)

        ptr_key = cgutils.alloca_once_value(builder, data_key)
        ptr_val = cgutils.alloca_once_value(builder, data_val)
        # TODO: the ptr_oldval is not used.  needed for refct
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
        fn = builder.module.get_or_insert_function(fnty, name='numba_dict_length')
        [d] = args
        [td] = sig.args
        dp = _dict_get_data(context, builder, td, d)
        n = builder.call(fn, [dp])
        return n

    return sig, codegen


@intrinsic
def _dict_dump(typingctx, d):
    """Dump the dictionary keys and values.
    Wraps numba_dict_dump for debugging.
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
        fn = builder.module.get_or_insert_function(fnty, name='numba_dict_dump')

        builder.call(fn, [dp])

    return sig, codegen


@intrinsic
def _dict_lookup(typingctx, d, key, hashval):
    """Wrap numba_dict_lookup

    Returns 2-tuple of (intp, ?value_type)
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
        fn = builder.module.get_or_insert_function(fnty, name='numba_dict_lookup')

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
        # Load value if output is available
        found = builder.icmp_signed('>=', ix, ix.type(int(DKIX.EMPTY)))

        out = context.make_optional_none(builder, td.value_type)
        pout = cgutils.alloca_once_value(builder, out)

        with builder.if_then(found):
            val = dm_val.load_from_data_pointer(builder, ptr_val)
            context.nrt.incref(builder, td.value_type, val)
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
        fn = builder.module.get_or_insert_function(fnty, name='numba_dict_popitem')

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

        cond = builder.icmp_signed('==', status, status.type(int(Status.OK)))
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

        fn = builder.module.get_or_insert_function(fnty, name='numba_dict_delitem')

        dp = _dict_get_data(context, builder, td, d)
        status = builder.call(fn, [dp, hk, ix])
        return status

    return sig, codegen


def _iterator_codegen(resty):
    """The common codegen for iterator intrinsics.

    Populates the iterator struct and increfs.
    """

    def codegen(context, builder, sig, args):
        [d] = args
        [td] = sig.args
        iterhelper = context.make_helper(builder, resty)
        iterhelper.parent = d
        iterhelper.state = iterhelper.state.type(None)
        return impl_ret_borrowed(
            context,
            builder,
            resty,
            iterhelper._getvalue(),
        )

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

        alloc_size = context.get_abi_sizeof(
            context.get_value_type(types.voidptr),
        )
        dtor = _imp_dtor(context, builder.module)
        meminfo = context.nrt.meminfo_alloc_dtor(
            builder,
            context.get_constant(types.uintp, alloc_size),
            dtor,
        )

        data_pointer = context.nrt.meminfo_data(builder, meminfo)
        data_pointer = builder.bitcast(data_pointer, ll_dict_type.as_pointer())
        builder.store(ptr, data_pointer)

        dstruct.meminfo = meminfo

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
        raise TypeError("expecting *key* and *value* to be a numba Type")

    keyty, valty = key, value

    def imp(key, value):
        dp = _dict_new_minsize(keyty, valty)
        _dict_set_method_table(dp, keyty, valty)
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
        return _dict_length(d)

    return impl


@overload(operator.setitem)
def impl_setitem(d, key, value):
    if not isinstance(d, types.DictType):
        return

    keyty, valty = d.key_type, d.value_type

    def impl(d, key, value):
        castedkey = _cast(key, keyty)
        castedval = _cast(value, valty)
        status = _dict_insert(d, castedkey, hash(castedkey), castedval)
        if status == Status.OK:
            return
        elif status == Status.OK_REPLACED:
            # replaced
            # XXX handle refcount
            return
        elif status == Status.ERR_CMP_FAILED:
            raise ValueError('key comparison failed')
        else:
            raise RuntimeError('dict.__setitem__ failed unexpectedly')
    return impl


@overload_method(types.DictType, 'get')
def impl_get(dct, key, default=None):
    if not isinstance(dct, types.DictType):
        return
    keyty = dct.key_type
    valty = dct.value_type
    _sentry_safe_cast_default(default, valty)

    def impl(dct, key, default=None):
        castedkey = _cast(key, keyty)
        ix, val = _dict_lookup(dct, key, hash(castedkey))
        if ix > DKIX.EMPTY:
            return val
        return default

    return impl


@overload(operator.getitem)
def impl_getitem(d, key):
    if not isinstance(d, types.DictType):
        return

    keyty = d.key_type

    def impl(d, key):
        castedkey = _cast(key, keyty)
        ix, val = _dict_lookup(d, castedkey, hash(castedkey))
        if ix == DKIX.EMPTY:
            raise KeyError()
        elif ix < DKIX.EMPTY:
            raise AssertionError("internal dict error during lookup")
        else:
            return _nonoptional(val)

    return impl


@overload_method(types.DictType, 'popitem')
def impl_popitem(d):
    if not isinstance(d, types.DictType):
        return

    def impl(d):
        status, keyval = _dict_popitem(d)
        if status == Status.OK:
            return _nonoptional(keyval)
        elif status == Status.ERR_DICT_EMPTY:
            raise KeyError()
        else:
            raise AssertionError('internal dict error during popitem')

    return impl


@overload_method(types.DictType, 'pop')
def impl_pop(dct, key, default=None):
    if not isinstance(dct, types.DictType):
        return

    keyty = dct.key_type
    valty = dct.value_type
    should_raise = isinstance(default, types.Omitted)
    _sentry_safe_cast_default(default, valty)

    def impl(dct, key, default=None):
        castedkey = _cast(key, keyty)
        hashed = hash(castedkey)
        ix, val = _dict_lookup(dct, castedkey, hashed)
        if ix == DKIX.EMPTY:
            if should_raise:
                raise KeyError()
            else:
                return default
        elif ix < DKIX.EMPTY:
            raise AssertionError("internal dict error during lookup")
        else:
            status = _dict_delitem(dct,hashed, ix)
            if status != Status.OK:
                raise AssertionError("internal dict error during delitem")
            return val

    return impl


@overload(operator.delitem)
def impl_delitem(d, k):
    if not isinstance(d, types.DictType):
        return

    def impl(d, k):
        d.pop(k)
    return impl


@overload(operator.contains)
def impl_contains(d, k):
    if not isinstance(d, types.DictType):
        return

    keyty = d.key_type

    def impl(d, k):
        k = _cast(k, keyty)
        ix, val = _dict_lookup(d, k, hash(k))
        return ix > DKIX.EMPTY
    return impl


@overload_method(types.DictType, 'clear')
def impl_clear(d):
    if not isinstance(d, types.DictType):
        return

    def impl(d):
        while len(d):
            d.popitem()

    return impl


@overload_method(types.DictType, 'copy')
def impl_copy(d):
    if not isinstance(d, types.DictType):
        return

    key_type, val_type = d.key_type, d.value_type

    def impl(d):
        newd = new_dict(key_type, val_type)
        for k, v in d.items():
            newd[k] = v
        return newd

    return impl


@overload_method(types.DictType, 'setdefault')
def impl_setdefault(dct, key, default=None):
    if not isinstance(dct, types.DictType):
        return

    def impl(dct, key, default=None):
        if key not in dct:
            dct[key] = default
        return dct[key]

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


@overload(operator.eq)
def impl_equal(da, db):
    if not isinstance(da, types.DictType):
        return
    if not isinstance(db, types.DictType):
        # If RHS is not a dictionary, always returns False
        def impl_type_mismatch(da, db):
            return False
        return impl_type_mismatch

    otherkeyty = db.key_type

    def impl_type_matched(da, db):
        if len(da) != len(db):
            return False
        for ka, va in da.items():
            # Cast key from LHS to the key-type of RHS
            kb = _cast(ka, otherkeyty)
            ix, vb = _dict_lookup(db, kb, hash(kb))
            if ix <= DKIX.EMPTY:
                # Quit early if the key is not found
                return False
            if va != vb:
                # Quit early if the values do not match
                return False
        return True

    return impl_type_matched


@overload(operator.ne)
def impl_not_equal(da, db):
    if not isinstance(da, types.DictType):
        return

    def impl(da, db):
        return not (da == db)

    return impl


@lower_builtin('getiter', types.DictItemsIterableType)
@lower_builtin('getiter', types.DictKeysIterableType)
@lower_builtin('getiter', types.DictValuesIterableType)
def impl_iterable_getiter(context, builder, sig, args):
    """Implement iter() for .keys(), .values(), .items()
    """
    iterablety = sig.args[0]
    it = context.make_helper(builder, iterablety.iterator_type, args[0])

    fnty = ir.FunctionType(
        ir.VoidType(),
        [ll_dictiter_type, ll_dict_type],
    )

    fn = builder.module.get_or_insert_function(fnty, name='numba_dict_iter')

    proto = ctypes.CFUNCTYPE(ctypes.c_size_t)
    dictiter_sizeof = proto(_helperlib.c_helpers['dict_iter_sizeof'])
    state_type = ir.ArrayType(ir.IntType(8), dictiter_sizeof())

    pstate = cgutils.alloca_once(builder, state_type, zfill=True)
    it.state = _as_bytes(builder, pstate)

    dp = _dict_get_data(context, builder, iterablety.parent, it.parent)
    builder.call(fn, [it.state, dp])
    return impl_ret_borrowed(
        context,
        builder,
        sig.return_type,
        it._getvalue(),
    )


@lower_builtin('getiter', types.DictType)
def impl_dict_getiter(context, builder, sig, args):
    """Implement iter(Dict).  Semantically equivalent to dict.keys()
    """
    [td] = sig.args
    [d] = args
    iterablety = types.DictKeysIterableType(td)
    it = context.make_helper(builder, iterablety.iterator_type)

    fnty = ir.FunctionType(
        ir.VoidType(),
        [ll_dictiter_type, ll_dict_type],
    )

    fn = builder.module.get_or_insert_function(fnty, name='numba_dict_iter')

    proto = ctypes.CFUNCTYPE(ctypes.c_size_t)
    dictiter_sizeof = proto(_helperlib.c_helpers['dict_iter_sizeof'])
    state_type = ir.ArrayType(ir.IntType(8), dictiter_sizeof())

    pstate = cgutils.alloca_once(builder, state_type, zfill=True)
    it.state = _as_bytes(builder, pstate)
    it.parent = d

    dp = _dict_get_data(context, builder, iterablety.parent, args[0])
    builder.call(fn, [it.state, dp])
    return impl_ret_borrowed(
        context,
        builder,
        sig.return_type,
        it._getvalue(),
    )


@lower_builtin('iternext', types.DictIteratorType)
@iternext_impl
def impl_iterator_iternext(context, builder, sig, args, result):
    iter_type = sig.args[0]
    it = context.make_helper(builder, iter_type, args[0])

    p2p_bytes = ll_bytes.as_pointer()

    iternext_fnty = ir.FunctionType(
        ll_status,
        [ll_bytes, p2p_bytes, p2p_bytes]
    )
    iternext = builder.module.get_or_insert_function(
        iternext_fnty,
        name='numba_dict_iter_next',
    )
    key_raw_ptr = cgutils.alloca_once(builder, ll_bytes)
    val_raw_ptr = cgutils.alloca_once(builder, ll_bytes)

    status = builder.call(iternext, (it.state, key_raw_ptr, val_raw_ptr))
    # TODO: no handling of error state i.e. mutated dictionary
    #       all errors are treated as exhausted iterator
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

        # All dict iterators use this common implementation.
        # Their differences are resolved here.
        if isinstance(iter_type.iterable, DictItemsIterableType):
            # .items()
            tup = context.make_tuple(builder, yield_type, [key, val])
            result.yield_(tup)
        elif isinstance(iter_type.iterable, DictKeysIterableType):
            # .keys()
            result.yield_(key)
        elif isinstance(iter_type.iterable, DictValuesIterableType):
            # .values()
            result.yield_(val)
        else:
            # unreachable
            raise AssertionError('unknown type: {}'.format(iter_type.iterable))
