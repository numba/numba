"""
Compiler-side implementation of the list.
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
    ListType,
    DictItemsIterableType,
    DictKeysIterableType,
    DictValuesIterableType,
    DictIteratorType,
    Type,
)
from numba.typeconv import Conversion
from numba.targets.imputils import impl_ret_borrowed, RefType
from numba.errors import TypingError
from numba import typing


ll_list_type = cgutils.voidptr_t
ll_listiter_type = cgutils.voidptr_t
ll_voidptr_type = cgutils.voidptr_t
ll_status = cgutils.int32_t
ll_ssize_t = cgutils.intp_t
ll_bytes = cgutils.voidptr_t


_meminfo_listptr = types.MemInfoPointer(types.voidptr)


@register_model(ListType)
class ListModel(models.StructModel):
    def __init__(self, dmm, fe_type):
        members = [
            ('meminfo', _meminfo_listptr),
            ('data', types.voidptr),   # ptr to the C list
        ]
        super(ListModel, self).__init__(dmm, fe_type, members)


class ListStatus(IntEnum):
    """Status code for other list operations.
    """
    LIST_OK = 0,
    LIST_ERR_INDEX = -1
    LIST_ERR_NO_MEMORY = -2


def _raise_if_error(context, builder, status, msg):
    """Raise an internal error depending on the value of *status*
    """
    ok_status = status.type(int(ListStatus.LIST_OK))
    with builder.if_then(builder.icmp_signed('!=', status, ok_status)):
        context.call_conv.return_user_exc(builder, RuntimeError, (msg,))


@intrinsic
def _as_meminfo(typingctx, lstobj):
    """Returns the MemInfoPointer of a list.
    """
    if not isinstance(lstobj, types.ListType):
        raise TypingError('expected *lstobj* to be a ListType')

    def codegen(context, builder, sig, args):
        [tl] = sig.args
        [l] = args
        # Incref
        context.nrt.incref(builder, tl, l)
        ctor = cgutils.create_struct_proxy(tl)
        lstruct = ctor(context, builder, value=l)
        # Returns the plain MemInfo
        return lstruct.meminfo

    sig = _meminfo_listptr(lstobj)
    return sig, codegen


def _list_get_data(context, builder, list_ty, l):
    """Helper to get the C list pointer in a numba list.
    """
    ctor = cgutils.create_struct_proxy(list_ty)
    lstruct = ctor(context, builder, value=l)
    return lstruct.data


# FIXME: copied from dictobject.py
def _as_bytes(builder, ptr):
    """Helper to do (void*)ptr
    """
    return builder.bitcast(ptr, cgutils.voidptr_t)


# FIXME: copied from dictobject.py
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


# FIXME: copied from dictobject.py
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


# FIXME: copied from dictobject.py
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


def _call_list_free(context, builder, ptr):
    """Call numba_list_free(ptr)
    """
    fnty = ir.FunctionType(
        ir.VoidType(),
        [ll_list_type],
    )
    free = builder.module.get_or_insert_function(fnty, name='numba_list_free')
    builder.call(free, [ptr])


# FIXME: this needs a careful review
def _imp_dtor(context, module):
    """Define the dtor for list
    """
    llvoidptr = context.get_value_type(types.voidptr)
    llsize = context.get_value_type(types.uintp)
    fnty = ir.FunctionType(
        ir.VoidType(),
        [llvoidptr, llsize, llvoidptr],
    )
    fname = '_numba_list_dtor'
    fn = module.get_or_insert_function(fnty, name=fname)

    if fn.is_declaration:
        # Set linkage
        fn.linkage = 'linkonce_odr'
        # Define
        builder = ir.IRBuilder(fn.append_basic_block())
        lp = builder.bitcast(fn.args[0], ll_list_type.as_pointer())
        l = builder.load(lp)
        _call_list_free(context, builder, l)
        builder.ret_void()

    return fn


def new_list(item):
    """Construct a new list. (Not implemented in the interpreter yet)

    Parameters
    ----------
    item: TypeRef
        Item type of the new list.
    """
    raise NotImplementedError


@intrinsic
def _make_list(typingctx, itemty, ptr):
    """Make a list struct with the given *ptr*

    Parameters
    ----------
    itemty: Type
        Type of the item.
    ptr : llvm pointer value
        Points to the list object.
    """
    list_ty = types.ListType(itemty.instance_type)

    def codegen(context, builder, signature, args):
        [_, ptr] = args
        ctor = cgutils.create_struct_proxy(list_ty)
        lstruct = ctor(context, builder)
        lstruct.data = ptr

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
        data_pointer = builder.bitcast(data_pointer, ll_list_type.as_pointer())
        builder.store(ptr, data_pointer)

        lstruct.meminfo = meminfo

        return lstruct._getvalue()

    sig = list_ty(itemty, ptr)
    return sig, codegen


@intrinsic
def _list_new(typingctx, itemty):
    """Wrap numba_list_new.

    Allocate a new list object with zero capacity.

    Parameters
    ----------
    itemty: Type
        Type of the items

    """
    resty = types.voidptr
    sig = resty(itemty)

    def codegen(context, builder, sig, args):
        fnty = ir.FunctionType(
            ll_status,
            [ll_list_type.as_pointer(), ll_ssize_t, ll_ssize_t],
        )
        fn = builder.module.get_or_insert_function(fnty, name='numba_list_new')
        # Determine sizeof item types
        ll_item = context.get_data_type(itemty.instance_type)
        sz_item = context.get_abi_sizeof(ll_item)
        reflp = cgutils.alloca_once(builder, ll_list_type, zfill=True)
        status = builder.call(
            fn,
            [reflp, ll_ssize_t(sz_item), ll_ssize_t(0)],
        )
        _raise_if_error(
            context, builder, status,
            msg="Failed to allocate list",
        )
        lp = builder.load(reflp)
        return lp

    return sig, codegen


@overload(new_list)
def impl_new_list(item):
    """Creates a new list with *item* as the type
    of the list item, respectively.
    """
    if not isinstance(item, Type):
        raise TypeError("expecting *item* to be a numba Type")

    itemty = item

    def imp(item):
        lp = _list_new(itemty)
        l = _make_list(itemty, lp)
        return l

    return imp


@overload(len)
def impl_len(l):
    """len(list)
    """
    if not isinstance(l, types.ListType):
        return

    def impl(l):
        return _list_length(l)

    return impl


@intrinsic
def _list_length(typingctx, l):
    """Wrap numba_list_length

    Returns the length of the list.
    """
    resty = types.intp
    sig = resty(l)

    def codegen(context, builder, sig, args):
        fnty = ir.FunctionType(
            ll_ssize_t,
            [ll_list_type],
        )
        fn = builder.module.get_or_insert_function(fnty, name='numba_list_length')
        [l] = args
        [tl] = sig.args
        lp = _list_get_data(context, builder, tl, l)
        n = builder.call(fn, [lp])
        return n

    return sig, codegen


@intrinsic
def _list_append(typingctx, l, item):
    """Wrap numba_list_append
    """
    resty = types.int32
    sig = resty(l, l.item_type)

    def codegen(context, builder, sig, args):
        fnty = ir.FunctionType(
            ll_status,
            [ll_list_type, ll_bytes],
        )
        [l, item] = args
        [tl, titem] = sig.args
        fn = builder.module.get_or_insert_function(fnty, name='numba_list_append')

        dm_item = context.data_model_manager[titem]

        data_item = dm_item.as_data(builder, item)

        ptr_item = cgutils.alloca_once_value(builder, data_item)

        lp = _list_get_data(context, builder, tl, l)
        status = builder.call(
            fn,
            [
                lp,
                _as_bytes(builder, ptr_item),
            ],
        )
        return status

    return sig, codegen


@overload_method(types.ListType, 'append')
def impl_append(l, item):
    if not isinstance(l, types.ListType):
        return

    itemty = l.item_type

    def impl(l, item):
        casteditem = _cast(item, itemty)
        status = _list_append(l, casteditem)
        if status == ListStatus.LIST_OK:
            return
        elif status == ListStatus.LIST_ERR_NO_MEMORY:
            raise MemoryError('Unable to allocate memory to append item')
        else:
            raise RuntimeError('list.append failed unexpectedly')

    if l.is_precise():
        # Handle the precise case.
        return impl
    else:
        # Handle the imprecise case.
        l = l.refine(item)
        # Re-bind the item type match the arguments.
        itemty = l.item_type
        # Create the signature that we wanted this impl to have.
        sig = typing.signature(types.void, l, itemty)
        return sig, impl


@intrinsic
def _list_getitem(typingctx, l, index):
    """Wrap numba_list_getitem

    Returns 2-tuple of (intp, ?item_type)
    """
    resty = types.Tuple([types.int32, types.Optional(l.item_type)])
    sig = resty(l, index)

    def codegen(context, builder, sig, args):
        fnty = ir.FunctionType(
            ll_status,
            [ll_list_type, ll_ssize_t, ll_bytes],
        )
        [tl, tindex] = sig.args
        [l, index] = args
        fn = builder.module.get_or_insert_function(fnty, name='numba_list_getitem')

        dm_item = context.data_model_manager[tl.item_type]
        ll_item = context.get_data_type(tl.item_type)
        ptr_item = cgutils.alloca_once(builder, ll_item)

        lp = _list_get_data(context, builder, tl, l)
        status = builder.call(
            fn,
            [
                lp,
                index,
                _as_bytes(builder, ptr_item),
            ],
        )
        # Load item if output is available
        found = builder.icmp_signed('>=', status, status.type(int(ListStatus.LIST_OK)))

        out = context.make_optional_none(builder, tl.item_type)
        pout = cgutils.alloca_once_value(builder, out)

        with builder.if_then(found):
            item = dm_item.load_from_data_pointer(builder, ptr_item)
            context.nrt.incref(builder, tl.item_type, item)
            loaded = context.make_optional_value(builder, tl.item_type, item)
            builder.store(loaded, pout)

        out = builder.load(pout)
        return context.make_tuple(builder, resty, [status, out])

    return sig, codegen


@overload(operator.getitem)
def impl_getitem(l, index):
    if not isinstance(l, types.ListType):
        return

    indexty = types.intp

    def impl(l, index):
        castedindex = _cast(index, indexty)
        status, item = _list_getitem(l, castedindex)
        if status == ListStatus.LIST_OK:
            return _nonoptional(item)
        elif status == ListStatus.LIST_ERR_INDEX:
            raise IndexError("list index out of range")
        else:
            raise AssertionError("internal list error during getitem")

    return impl


@intrinsic
def _list_setitem(typingctx, l, index, item):
    """Wrap numba_list_setitem
    """
    resty = types.int32
    sig = resty(l, index, item)

    def codegen(context, builder, sig, args):
        fnty = ir.FunctionType(
            ll_status,
            [ll_list_type, ll_ssize_t, ll_bytes],
        )
        [l, index, item] = args
        [tl, tindex, titem] = sig.args
        fn = builder.module.get_or_insert_function(fnty,
                                                   name='numba_list_setitem')

        dm_item = context.data_model_manager[titem]
        data_item = dm_item.as_data(builder, item)
        ptr_item = cgutils.alloca_once_value(builder, data_item)

        lp = _list_get_data(context, builder, tl, l)
        status = builder.call(
            fn,
            [
                lp,
                index,
                _as_bytes(builder, ptr_item),
            ],
        )
        return status

    return sig, codegen


@overload(operator.setitem)
def impl_setitem(l, index, item):
    if not isinstance(l, types.ListType):
        return

    indexty = types.intp
    itemty = l.item_type

    def impl(l, index, item):
        castedindex = _cast(index, indexty)
        casteditem = _cast(item, itemty)
        status = _list_setitem(l, castedindex, casteditem)
        if status == ListStatus.LIST_OK:
            pass
        elif status == ListStatus.LIST_ERR_INDEX:
            raise IndexError("list index out of range")
        else:
            raise AssertionError("internal list error during settitem")

    return impl
