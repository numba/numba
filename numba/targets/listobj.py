"""
"""

from __future__ import print_function, absolute_import, division

import math

from llvmlite import ir
from numba import types, cgutils, typing
from numba.targets.imputils import (builtin, builtin_attr, implement,
                                    impl_attribute, impl_attribute_generic,
                                    iternext_impl, struct_factory,
                                    impl_ret_borrowed, impl_ret_new_ref,
                                    impl_ret_untracked)


def make_list_cls(list_type):
    """
    Return the Structure representation of the given *list_type*
    (an instance of types.List).
    """
    return cgutils.create_struct_proxy(list_type)


def get_itemsize(context, list_type):
    """
    Return the data type and item size for the given list type.
    """
    llty = context.get_data_type(list_type.dtype)
    return llty, context.get_abi_sizeof(llty)


def _build_list_uninitialized(context, builder, list_type, nitems):
    """
    Make a list structure with allocated data.
    """
    listcls = make_list_cls(list_type)
    list = listcls(context, builder)

    intp_t = context.get_value_type(types.intp)

    datatype, itemsize = get_itemsize(context, list_type)
    
    allocsize = ir.Constant(intp_t, nitems * itemsize)

    meminfo = context.nrt_meminfo_alloc_aligned(builder, size=allocsize,
                                                align=32)
    data = context.nrt_meminfo_data(builder, meminfo)
    data = builder.bitcast(data, datatype.as_pointer())
    # XXX handle allocation failure

    size = ir.Constant(intp_t, nitems)
    list.meminfo = meminfo
    list.size = cgutils.alloca_once_value(builder, size)
    list.allocated = cgutils.alloca_once_value(builder, size)
    list.data = cgutils.alloca_once_value(builder, data)
    return list


def _list_setitem(context, builder, list, idx, val):
    # XXX NRT?
    ptr = cgutils.gep(builder, builder.load(list.data), idx)
    builder.store(val, ptr)


def build_list(context, builder, list_type, items):
    list = _build_list_uninitialized(context, builder, list_type, len(items))
    for i, val in enumerate(items):
        _list_setitem(context, builder, list, i, val)

    return impl_ret_new_ref(context, builder, list_type, list._getvalue())


#-------------------------------------------------------------------------------
# Various operations

@builtin
@implement(types.len_type, types.Kind(types.List))
def list_len(context, builder, sig, args):
    (list_type,) = sig.args
    (list_val,) = args
    list = make_list_cls(list_type)(context, builder, list_val)
    list_len = builder.load(list.size)
    return impl_ret_untracked(context, builder, sig.return_type, list_len)


@struct_factory(types.ListIter)
def make_listiter_cls(iterator_type):
    """
    Return the Structure representation of the given *iterator_type* (an
    instance of types.ListIter).
    """
    return cgutils.create_struct_proxy(iterator_type)

@builtin
@implement('getiter', types.Kind(types.List))
def getiter_array(context, builder, sig, args):
    (list_type,) = sig.args
    (list_val,) = args

    iterobj = make_listiter_cls(sig.return_type)(context, builder)
    list = make_list_cls(list_type)(context, builder, list_val)

    index = context.get_constant(types.intp, 0)
    iterobj.index = cgutils.alloca_once_value(builder, index)
    iterobj.size = cgutils.alloca_once_value(builder, builder.load(list.size))
    iterobj.data = cgutils.alloca_once_value(builder, builder.load(list.data))

    # XXX? Incref array
    #if context.enable_nrt:
        #context.nrt_incref(builder, list_type, list_val)

    # Note: a decref on the iterator will dereference all internal MemInfo*
    return iterobj._getvalue()
    out = impl_ret_new_ref(context, builder, sig.return_type, iterobj._getvalue())
    return out

@builtin
@implement('iternext', types.Kind(types.ListIter))
@iternext_impl
def iternext_array(context, builder, sig, args, result):
    (iter_type,) = sig.args
    (iter_val,) = args
    list_type = iter_type.list

    iterobj = make_listiter_cls(iter_type)(context, builder, iter_val)

    index = builder.load(iterobj.index)
    nitems = builder.load(iterobj.size)
    is_valid = builder.icmp_signed('<', index, nitems)
    result.set_valid(is_valid)

    with builder.if_then(is_valid):
        ptr = cgutils.gep(builder, builder.load(iterobj.data), index)
        value = builder.load(ptr)
        result.yield_(value)
        nindex = builder.add(index, context.get_constant(types.intp, 1))
        builder.store(nindex, iterobj.index)
