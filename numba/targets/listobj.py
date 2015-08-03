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


def make_payload_cls(list_type):
    """
    Return the Structure representation of the given *list_type*'s payload
    (an instance of types.List).
    """
    return cgutils.create_struct_proxy(types.ListPayload(list_type))


def make_payload_obj(context, builder, list_type, payload):
    """
    Make a payload Structure from a payload pointer.
    """
    payload_type = context.get_data_type(types.ListPayload(list_type))
    payload = builder.bitcast(payload, payload_type.as_pointer())
    return make_payload_cls(list_type)(context, builder, ref=payload)


def get_list_payload(context, builder, list_type, value):
    """
    Given a list value and type, get its payload structure (as a
    reference, so that mutations are seen by all).
    """
    payload_type = context.get_data_type(types.ListPayload(list_type))
    payload = context.nrt_meminfo_data(builder, value.meminfo)
    payload = builder.bitcast(payload, payload_type.as_pointer())
    return make_payload_cls(list_type)(context, builder, ref=payload)


def get_itemsize(context, list_type):
    """
    Return the item size for the given list type.
    """
    llty = context.get_data_type(list_type.dtype)
    return context.get_abi_sizeof(llty)


def _build_list_uninitialized(context, builder, list_type, nitems):
    """
    Make a list structure and payload with allocated data for *nitems*.
    """
    intp_t = context.get_value_type(types.intp)

    payload_type = context.get_data_type(types.ListPayload(list_type))
    payload_size = context.get_abi_sizeof(payload_type)

    itemsize = get_itemsize(context, list_type)
    
    allocsize = ir.Constant(intp_t, payload_size + nitems * itemsize)
    # XXX aligned? need cooperation with realloc
    meminfo = context.nrt_meminfo_varsize_alloc(builder, size=allocsize)
    # XXX handle allocation failure

    list_obj = make_list_cls(list_type)(context, builder)
    size = ir.Constant(intp_t, nitems)
    list_obj.meminfo = meminfo
    payload_obj = get_list_payload(context, builder, list_type, list_obj)
    payload_obj.allocated = size
    return list_obj, payload_obj


def _payload_getitem(context, builder, payload, idx):
    ptr = cgutils.gep(builder, payload._get_ptr_by_name('data'), idx)
    return builder.load(ptr)

def _payload_setitem(context, builder, payload, idx, val):
    # XXX NRT?
    ptr = cgutils.gep(builder, payload._get_ptr_by_name('data'), idx)
    builder.store(val, ptr)

def _payload_realloc(context, builder, list_type, list_obj, new_allocated):
    payload_type = context.get_data_type(types.ListPayload(list_type))
    payload_size = context.get_abi_sizeof(payload_type)

    itemsize = get_itemsize(context, list_type)

    allocsize = builder.mul(ir.Constant(new_allocated.type, itemsize),
                            new_allocated)
    allocsize = builder.add(ir.Constant(new_allocated.type, payload_size),
                            allocsize)
    payload = context.nrt_meminfo_varsize_realloc(builder, list_obj.meminfo,
                                                  size=allocsize)
    payload_obj = make_payload_obj(context, builder, list_type, payload)
    # XXX handle allocation failure
    payload_obj.allocated = new_allocated


def build_list(context, builder, list_type, items):
    """
    Build a list of the given type, containing the given items.
    """
    nitems = len(items)
    list_obj, payload = _build_list_uninitialized(context, builder,
                                                  list_type, nitems)
    # Populate list
    payload.size = context.get_constant(types.intp, nitems)
    for i, val in enumerate(items):
        _payload_setitem(context, builder, payload, i, val)

    return impl_ret_new_ref(context, builder, list_type, list_obj._getvalue())

def resize_list(context, builder, list_type, list_obj, new_size):
    """
    Ensure the list is properly sized for the new size.
    """
    payload_obj = get_list_payload(context, builder, list_type, list_obj)
    itemsize = get_itemsize(context, list_type)

    allocated = payload_obj.allocated

    one = ir.Constant(new_size.type, 1)
    two = ir.Constant(new_size.type, 2)
    eight = ir.Constant(new_size.type, 8)

    # allocated < new_size
    is_too_small = builder.icmp_signed('<', allocated, new_size)
    # (allocated >> 2) > new_size
    is_too_large = builder.icmp_signed('>', builder.ashr(allocated, two), new_size)
    
    with builder.if_then(is_too_large, likely=False):
        # Exact downsize to requested size
        # NOTE: is_too_large must be aggressive enough to avoid repeated
        # upsizes and downsizes when growing a list.
        _payload_realloc(context, builder, list_type, list_obj, new_size)
    
    with builder.if_then(is_too_small, likely=False):
        # Upsize with moderate over-allocation (size + size >> 2 + 8)
        new_allocated = builder.add(eight,
                                    builder.add(new_size,
                                                builder.ashr(new_size, two)))
        _payload_realloc(context, builder, list_type, list_obj, new_allocated)

    # The payload's address may have changed
    payload_obj = get_list_payload(context, builder, list_type, list_obj)
    payload_obj.size = new_size
    return payload_obj


#-------------------------------------------------------------------------------
# Various operations

@builtin
@implement(types.len_type, types.Kind(types.List))
def list_len(context, builder, sig, args):
    (list_type,) = sig.args
    (list_val,) = args
    list = make_list_cls(list_type)(context, builder, list_val)
    payload = get_list_payload(context, builder, list_type, list)
    return impl_ret_untracked(context, builder, sig.return_type, payload.size)


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
    list_obj = make_list_cls(list_type)(context, builder, list_val)

    index = context.get_constant(types.intp, 0)
    iterobj.index = cgutils.alloca_once_value(builder, index)
    iterobj.meminfo = list_obj.meminfo

    # The iterator shares its meminfo with the array, so is currently borrowed
    out = impl_ret_borrowed(context, builder, sig.return_type, iterobj._getvalue())
    return out

@builtin
@implement('iternext', types.Kind(types.ListIter))
@iternext_impl
def iternext_array(context, builder, sig, args, result):
    (iter_type,) = sig.args
    (iter_val,) = args
    list_type = iter_type.list

    iterobj = make_listiter_cls(iter_type)(context, builder, iter_val)
    payload = get_list_payload(context, builder, list_type, iterobj)

    index = builder.load(iterobj.index)
    nitems = payload.size
    is_valid = builder.icmp_signed('<', index, nitems)
    result.set_valid(is_valid)

    with builder.if_then(is_valid):
        result.yield_(_payload_getitem(context, builder, payload, index))
        nindex = builder.add(index, context.get_constant(types.intp, 1))
        builder.store(nindex, iterobj.index)



#-------------------------------------------------------------------------------
# Methods

@builtin
@implement("list.pop", types.Kind(types.List))
def list_pop(context, builder, sig, args):
    list_type = sig.args[0]
    list_obj = make_list_cls(list_type)(context, builder, value=args[0])

    payload = get_list_payload(context, builder, list_type, list_obj)

    n = payload.size
    cgutils.guard_zero(context, builder, n,
                       (IndexError, "list index out of range"))

    n = builder.sub(n, ir.Constant(n.type, 1))
    res = _payload_getitem(context, builder, payload, n)
    payload = resize_list(context, builder, list_type, list_obj, n)
    payload.size = n
    return res


@builtin
@implement("list.append", types.Kind(types.List), types.Any)
def list_append(context, builder, sig, args):
    list_type = sig.args[0]
    list_obj = make_list_cls(list_type)(context, builder, value=args[0])
    item = args[1]

    payload = get_list_payload(context, builder, list_type, list_obj)

    n = payload.size
    new_size = builder.add(n, ir.Constant(n.type, 1))

    payload = resize_list(context, builder, list_type, list_obj, new_size)
    _payload_setitem(context, builder, payload, n, item)
    return context.get_dummy_value()

