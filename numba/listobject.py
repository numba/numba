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
            ('data', types.voidptr),   # ptr to the C dict
        ]
        super(ListModel, self).__init__(dmm, fe_type, members)


class ListStatus(IntEnum):
    """Status code for other list operations.
    """
    LIST_OK = 0,
    LIST_ERR_NO_MEMORY = -1


def _raise_if_error(context, builder, status, msg):
    """Raise an internal error depending on the value of *status*
    """
    ok_status = status.type(int(ListStatus.LIST_OK))
    with builder.if_then(builder.icmp_signed('!=', status, ok_status)):
        context.call_conv.return_user_exc(builder, RuntimeError, (msg,))


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
def _list_new(typingctx, itemty):
    """Wrap numba_list_new.

    Allocate a new list object with zero capacity.

    Parameters
    ----------
    itemty: Type
        Type of the key and value, respectively.

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
    of the list key item, respectively.
    """
    if not isinstance(item, Type):
        raise TypeError("expecting *item* and *value* to be a numba Type")

    itemty = item

    def imp(item):
        l = _list_new(itemty)
        return l

    return imp
