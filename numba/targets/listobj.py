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
from numba.utils import cached_property


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


class _ListPayloadMixin(object):

    @property
    def size(self):
        return self._payload.size

    @size.setter
    def size(self, value):
        self._payload.size = value

    @property
    def data(self):
        return self._payload._get_ptr_by_name('data')

    def _gep(self, idx):
        return cgutils.gep(self._builder, self.data, idx)

    # Note about NRT: lists of NRT-managed objects (included nested lists)
    # cannot be handled right now, as the number of nested meminfos is
    # dynamic.

    def getitem(self, idx):
        ptr = self._gep(idx)
        return self._builder.load(ptr)

    def setitem(self, idx, val):
        ptr = self._gep(idx)
        self._builder.store(val, ptr)

    def inititem(self, idx, val):
        ptr = self._gep(idx)
        self._builder.store(val, ptr)


class ListInstance(_ListPayloadMixin):
    
    def __init__(self, context, builder, list_type, list_val):
        self._context = context
        self._builder = builder
        self._ty = list_type
        self._list = make_list_cls(list_type)(context, builder, list_val)

    @property
    def _payload(self):
        # This cannot be cached as it can be reallocated
        return get_list_payload(self._context, self._builder, self._ty, self._list)

    @property
    def value(self):
        return self._list._getvalue()

    @property
    def meminfo(self):
        return self._list.meminfo

    @classmethod
    def allocate(cls, context, builder, list_type, nitems):
        intp_t = context.get_value_type(types.intp)

        payload_type = context.get_data_type(types.ListPayload(list_type))
        payload_size = context.get_abi_sizeof(payload_type)

        itemsize = get_itemsize(context, list_type)
        
        allocsize = ir.Constant(intp_t, payload_size + nitems * itemsize)
        meminfo = context.nrt_meminfo_varsize_alloc(builder, size=allocsize)
        # XXX handle allocation failure

        self = cls(context, builder, list_type, None)
        size = ir.Constant(intp_t, nitems)
        self._list.meminfo = meminfo
        self._payload.allocated = size
        return self

    def resize(self, new_size):
        """
        Ensure the list is properly sized for the new size.
        """
        def _payload_realloc(new_allocated):
            payload_type = context.get_data_type(types.ListPayload(self._ty))
            payload_size = context.get_abi_sizeof(payload_type)

            allocsize = builder.mul(ir.Constant(new_allocated.type, itemsize),
                                    new_allocated)
            allocsize = builder.add(ir.Constant(new_allocated.type, payload_size),
                                    allocsize)
            ptr = context.nrt_meminfo_varsize_realloc(builder, self._list.meminfo,
                                                      size=allocsize)
            # XXX handle allocation failure
            self._payload.allocated = new_allocated

        context = self._context
        builder = self._builder

        itemsize = get_itemsize(context, self._ty)
        allocated = self._payload.allocated

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
            _payload_realloc(new_size)

        with builder.if_then(is_too_small, likely=False):
            # Upsize with moderate over-allocation (size + size >> 2 + 8)
            new_allocated = builder.add(eight,
                                        builder.add(new_size,
                                                    builder.ashr(new_size, two)))
            _payload_realloc(new_allocated)

        self._payload.size = new_size


class ListIterInstance(_ListPayloadMixin):
    
    def __init__(self, context, builder, iter_type, iter_val):
        self._context = context
        self._builder = builder
        self._ty = iter_type
        self._iter = make_listiter_cls(iter_type)(context, builder, iter_val)

    @classmethod
    def from_list(cls, context, builder, iter_type, list_val):
        list_inst = ListInstance(context, builder, iter_type.list_type, list_val)
        self = cls(context, builder, iter_type, None)
        index = context.get_constant(types.intp, 0)
        self._iter.index = cgutils.alloca_once_value(builder, index)
        self._iter.meminfo = list_inst.meminfo
        return self

    @property
    def _payload(self):
        # This cannot be cached as it can be reallocated
        return get_list_payload(self._context, self._builder,
                                self._ty.list_type, self._iter)

    @property
    def value(self):
        return self._iter._getvalue()

    @property
    def index(self):
        return self._builder.load(self._iter.index)

    @index.setter
    def index(self, value):
        self._builder.store(value, self._iter.index)


#-------------------------------------------------------------------------------
# Constructors

def build_list(context, builder, list_type, items):
    """
    Build a list of the given type, containing the given items.
    """
    nitems = len(items)
    inst = ListInstance.allocate(context, builder, list_type, nitems)
    # Populate list
    inst.size = context.get_constant(types.intp, nitems)
    for i, val in enumerate(items):
        inst.setitem(context.get_constant(types.intp, i), val)

    return impl_ret_new_ref(context, builder, list_type, inst.value)


@builtin
@implement(list, types.Kind(types.IterableType))
def list_constructor(context, builder, sig, args):
    
    def list_impl(iterable):
        res = []
        for v in iterable:
            res.append(v)
        return res

    return context.compile_internal(builder, list_impl, sig, args)


#-------------------------------------------------------------------------------
# Various operations

@builtin
@implement(types.len_type, types.Kind(types.List))
def list_len(context, builder, sig, args):
    inst = ListInstance(context, builder, sig.args[0], args[0])
    return inst.size


@struct_factory(types.ListIter)
def make_listiter_cls(iterator_type):
    """
    Return the Structure representation of the given *iterator_type* (an
    instance of types.ListIter).
    """
    return cgutils.create_struct_proxy(iterator_type)

@builtin
@implement('getiter', types.Kind(types.List))
def getiter_list(context, builder, sig, args):
    inst = ListIterInstance.from_list(context, builder, sig.return_type, args[0])
    return impl_ret_borrowed(context, builder, sig.return_type, inst.value)

@builtin
@implement('iternext', types.Kind(types.ListIter))
@iternext_impl
def iternext_listiter(context, builder, sig, args, result):
    inst = ListIterInstance(context, builder, sig.args[0], args[0])

    index = inst.index
    nitems = inst.size
    is_valid = builder.icmp_signed('<', index, nitems)
    result.set_valid(is_valid)

    with builder.if_then(is_valid):
        result.yield_(inst.getitem(index))
        inst.index = builder.add(index, context.get_constant(types.intp, 1))


@builtin
@implement('getitem', types.Kind(types.List), types.Kind(types.Integer))
def getitem_list(context, builder, sig, args):
    inst = ListInstance(context, builder, sig.args[0], args[0])
    index = args[1]

    is_negative = builder.icmp_signed('<', index, ir.Constant(index.type, 0))
    wrapped_index = builder.add(index, inst.size)
    index = builder.select(is_negative, wrapped_index, index)

    result = inst.getitem(index)

    return impl_ret_borrowed(context, builder, sig.return_type, result)

@builtin
@implement('setitem', types.Kind(types.List), types.Kind(types.Integer), types.Any)
def setitem_list(context, builder, sig, args):
    inst = ListInstance(context, builder, sig.args[0], args[0])
    index = args[1]
    value = args[2]

    is_negative = builder.icmp_signed('<', index, ir.Constant(index.type, 0))
    wrapped_index = builder.add(index, inst.size)
    index = builder.select(is_negative, wrapped_index, index)

    inst.setitem(index, value)
    return context.get_dummy_value()


#-------------------------------------------------------------------------------
# Methods

@builtin
@implement("list.pop", types.Kind(types.List))
def list_pop(context, builder, sig, args):
    inst = ListInstance(context, builder, sig.args[0], args[0])

    n = inst.size
    cgutils.guard_zero(context, builder, n,
                       (IndexError, "list index out of range"))
    n = builder.sub(n, ir.Constant(n.type, 1))
    res = inst.getitem(n)
    inst.resize(n)
    return res


@builtin
@implement("list.append", types.Kind(types.List), types.Any)
def list_append(context, builder, sig, args):
    inst = ListInstance(context, builder, sig.args[0], args[0])
    item = args[1]

    n = inst.size
    new_size = builder.add(n, ir.Constant(n.type, 1))
    inst.resize(new_size)
    inst.setitem(n, item)

    return context.get_dummy_value()
