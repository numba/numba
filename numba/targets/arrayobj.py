"""
Implementation of operations on Array objects and objects supporting
the buffer protocol.
"""

from __future__ import print_function, absolute_import, division

import math

import llvmlite.llvmpy.core as lc
from llvmlite.llvmpy.core import Constant

import numpy
from numba import types, cgutils, typing
from numba.numpy_support import as_dtype
from numba.numpy_support import version as numpy_version
from numba.targets.imputils import (builtin, builtin_attr, implement,
                                    impl_attribute, impl_attribute_generic,
                                    iternext_impl, impl_ret_borrowed,
                                    impl_ret_new_ref, impl_ret_untracked)
from numba.typing import signature
from . import quicksort, slicing


def increment_index(builder, val):
    """
    Increment an index *val*.
    """
    one = Constant.int(val.type, 1)
    # We pass the "nsw" flag in the hope that LLVM understands the index
    # never changes sign.  Unfortunately this doesn't always work
    # (e.g. ndindex()).
    return builder.add(val, one, flags=['nsw'])


def set_range_metadata(builder, load, lower_bound, upper_bound):
    """
    Set the "range" metadata on a load instruction.
    Note the interval is in the form [lower_bound, upper_bound).
    """
    range_operands = [Constant.int(load.type, lower_bound),
                      Constant.int(load.type, upper_bound)]
    md = builder.module.add_metadata(range_operands)
    load.set_metadata("range", md)


def mark_positive(builder, load):
    """
    Mark the result of a load instruction as positive (or zero).
    """
    upper_bound = (1 << (load.type.width - 1)) - 1
    set_range_metadata(builder, load, 0, upper_bound)


def make_array(array_type):
    """
    Return the Structure representation of the given *array_type*
    (an instance of types.Array).
    """
    base = cgutils.create_struct_proxy(array_type)
    ndim = array_type.ndim

    class ArrayStruct(base):
        @property
        def shape(self):
            """
            Override .shape to inform LLVM that its elements are all positive.
            """
            builder = self._builder
            if ndim == 0:
                return base.__getattr__(self, "shape")

            # Unfortunately, we can't use llvm.assume as its presence can
            # seriously pessimize performance,
            # *and* the range metadata currently isn't improving anything here,
            # see https://llvm.org/bugs/show_bug.cgi?id=23848 !
            ptr = self._get_ptr_by_name("shape")
            dims = []
            for i in range(ndim):
                dimptr = cgutils.gep_inbounds(builder, ptr, 0, i)
                load = builder.load(dimptr)
                dims.append(load)
                mark_positive(builder, load)

            return cgutils.pack_array(builder, dims)

    return ArrayStruct


def get_itemsize(context, array_type):
    """
    Return the item size for the given array or buffer type.
    """
    llty = context.get_data_type(array_type.dtype)
    return context.get_abi_sizeof(llty)


def load_item(context, builder, arrayty, ptr):
    """
    Load the item at the given array pointer.
    """
    align = None if arrayty.aligned else 1
    return context.unpack_value(builder, arrayty.dtype, ptr,
                                align=align)

def store_item(context, builder, arrayty, val, ptr):
    """
    Store the item at the given array pointer.
    """
    align = None if arrayty.aligned else 1
    return context.pack_value(builder, arrayty.dtype, val, ptr, align=align)


def fix_integer_index(context, builder, idxty, idx, size):
    """
    Fix the integer index' type and value for the given dimension size.
    """
    if idxty.signed:
        ind = context.cast(builder, idx, idxty, types.intp)
        ind = slicing.fix_index(builder, ind, size)
    else:
        ind = context.cast(builder, idx, idxty, types.uintp)
    return ind


def populate_array(array, data, shape, strides, itemsize, meminfo,
                   parent=None):
    """
    Helper function for populating array structures.
    This avoids forgetting to set fields.

    *shape* and *strides* can be Python tuples or LLVM arrays.
    """
    context = array._context
    builder = array._builder
    datamodel = array._datamodel
    required_fields = set(datamodel._fields)

    if meminfo is None:
        meminfo = Constant.null(context.get_value_type(
            datamodel.get_type('meminfo')))

    intp_t = context.get_value_type(types.intp)
    if isinstance(shape, (tuple, list)):
        shape = cgutils.pack_array(builder, shape, intp_t)
    if isinstance(strides, (tuple, list)):
        strides = cgutils.pack_array(builder, strides, intp_t)

    attrs = dict(shape=shape,
                 strides=strides,
                 data=data,
                 itemsize=itemsize,
                 meminfo=meminfo,)

    # Set `parent` attribute
    if parent is None:
        attrs['parent'] = Constant.null(context.get_value_type(
            datamodel.get_type('parent')))
    else:
        attrs['parent'] = parent
    # Calc num of items from shape
    nitems = context.get_constant(types.intp, 1)
    unpacked_shape = cgutils.unpack_tuple(builder, shape, shape.type.count)
    if unpacked_shape:
        # Shape is not empty
        for axlen in unpacked_shape:
            nitems = builder.mul(nitems, axlen)
    else:
        # Shape is empty
        nitems = context.get_constant(types.intp, 1)
    attrs['nitems'] = nitems

    # Make sure that we have all the fields
    got_fields = set(attrs.keys())
    if got_fields != required_fields:
        raise ValueError("missing {0}".format(required_fields - got_fields))

    # Set field value
    for k, v in attrs.items():
        setattr(array, k, v)

    return array


def update_array_info(aryty, array):
    """
    Update some auxiliary information in *array* after some of its fields
    were changed.  `itemsize` and `nitems` are updated.
    """
    context = array._context
    builder = array._builder

    # Calc num of items from shape
    nitems = context.get_constant(types.intp, 1)
    unpacked_shape = cgutils.unpack_tuple(builder, array.shape, aryty.ndim)
    for axlen in unpacked_shape:
        nitems = builder.mul(nitems, axlen)
    array.nitems = nitems

    array.itemsize = context.get_constant(types.intp,
                                          get_itemsize(context, aryty))


def make_arrayiter_cls(iterator_type):
    """
    Return the Structure representation of the given *iterator_type* (an
    instance of types.ArrayIteratorType).
    """
    return cgutils.create_struct_proxy(iterator_type)

@builtin
@implement('getiter', types.Kind(types.Buffer))
def getiter_array(context, builder, sig, args):
    [arrayty] = sig.args
    [array] = args

    iterobj = make_arrayiter_cls(sig.return_type)(context, builder)

    zero = context.get_constant(types.intp, 0)
    indexptr = cgutils.alloca_once_value(builder, zero)

    iterobj.index = indexptr
    iterobj.array = array

    # Incref array
    if context.enable_nrt:
        context.nrt_incref(builder, arrayty, array)

    res = iterobj._getvalue()

    # Note: a decref on the iterator will dereference all internal MemInfo*
    out = impl_ret_new_ref(context, builder, sig.return_type, res)
    return out


def _getitem_array1d(context, builder, arrayty, array, idx, wraparound):
    """
    Look up and return an element from a 1D array.
    """
    ptr = cgutils.get_item_pointer(builder, arrayty, array, [idx],
                                   wraparound=wraparound)
    return load_item(context, builder, arrayty, ptr)

@builtin
@implement('iternext', types.Kind(types.ArrayIterator))
@iternext_impl
def iternext_array(context, builder, sig, args, result):
    [iterty] = sig.args
    [iter] = args
    arrayty = iterty.array_type

    if arrayty.ndim != 1:
        # TODO
        raise NotImplementedError("iterating over %dD array" % arrayty.ndim)

    iterobj = make_arrayiter_cls(iterty)(context, builder, value=iter)
    ary = make_array(arrayty)(context, builder, value=iterobj.array)

    nitems, = cgutils.unpack_tuple(builder, ary.shape, count=1)

    index = builder.load(iterobj.index)
    is_valid = builder.icmp(lc.ICMP_SLT, index, nitems)
    result.set_valid(is_valid)

    with builder.if_then(is_valid):
        value = _getitem_array1d(context, builder, arrayty, ary, index,
                                 wraparound=False)
        result.yield_(value)
        nindex = builder.add(index, context.get_constant(types.intp, 1))
        builder.store(nindex, iterobj.index)


#-------------------------------------------------------------------------------
# Basic indexing (with integers and slices only)

def basic_indexing(context, builder, aryty, ary, index_types, indices):
    """
    Perform basic indexing on the given array.
    A (data pointer, shapes, strides) tuple is returned describing
    the corresponding view.
    """
    zero = context.get_constant(types.intp, 0)

    shapes = cgutils.unpack_tuple(builder, ary.shape, aryty.ndim)
    strides = cgutils.unpack_tuple(builder, ary.strides, aryty.ndim)

    output_indices = []
    output_shapes = []
    output_strides = []

    ax = 0
    for indexval, idxty in zip(indices, index_types):
        if idxty is types.ellipsis:
            # Fill up missing dimensions at the middle
            n_missing = aryty.ndim - len(indices) + 1
            for i in range(n_missing):
                output_indices.append(zero)
                output_shapes.append(shapes[ax])
                output_strides.append(strides[ax])
                ax += 1
            continue
        # Regular index value
        if idxty == types.slice3_type:
            slice = slicing.Slice(context, builder, value=indexval)
            cgutils.guard_invalid_slice(context, builder, slice)
            slicing.fix_slice(builder, slice, shapes[ax])
            output_indices.append(slice.start)
            sh = slicing.get_slice_length(builder, slice)
            st = slicing.fix_stride(builder, slice, strides[ax])
            output_shapes.append(sh)
            output_strides.append(st)
        elif isinstance(idxty, types.Integer):
            ind = fix_integer_index(context, builder, idxty, indexval,
                                    shapes[ax])
            output_indices.append(ind)
        else:
            raise NotImplementedError("unexpected index type: %s" % (idxty,))
        ax += 1

    # Fill up missing dimensions at the end
    assert ax <= aryty.ndim
    while ax < aryty.ndim:
        output_shapes.append(shapes[ax])
        output_strides.append(strides[ax])
        ax += 1

    # No need to check wraparound, as negative indices were already
    # fixed in the loop above.
    dataptr = cgutils.get_item_pointer(builder, aryty, ary,
                                       output_indices,
                                       wraparound=False)
    return (dataptr, output_shapes, output_strides)


def make_view(context, builder, aryty, ary, return_type,
              data, shapes, strides):
    """
    Build a view over the given array with the given parameters.
    """
    retary = make_array(return_type)(context, builder)
    populate_array(retary,
                   data=data,
                   shape=shapes,
                   strides=strides,
                   itemsize=ary.itemsize,
                   meminfo=ary.meminfo,
                   parent=ary.parent)
    return retary


def _getitem_array_generic(context, builder, return_type, aryty, ary,
                           index_types, indices):
    """
    Return the result of indexing *ary* with the given *indices*.
    """
    assert isinstance(return_type, types.Buffer)
    dataptr, view_shapes, view_strides = \
        basic_indexing(context, builder, aryty, ary, index_types, indices)

    # Build array view
    retary = make_view(context, builder, aryty, ary, return_type,
                       dataptr, view_shapes, view_strides)
    return retary._getvalue()


@builtin
@implement('getitem', types.Kind(types.Buffer), types.Kind(types.Integer))
def getitem_arraynd_intp(context, builder, sig, args):
    aryty, idxty = sig.args
    ary, idx = args
    ary = make_array(aryty)(context, builder, ary)

    dataptr, shapes, strides = \
        basic_indexing(context, builder, aryty, ary, (idxty,), (idx,))

    ndim = aryty.ndim
    if ndim == 1:
        # Return a value
        assert not shapes
        result = load_item(context, builder, aryty, dataptr)
    elif ndim > 1:
        # Return a subview over the array
        out_ary = make_view(context, builder, aryty, ary, sig.return_type,
                            dataptr, shapes, strides)
        result = out_ary._getvalue()
    else:
        raise NotImplementedError("1D indexing into %dD array" % aryty.ndim)
    return impl_ret_borrowed(context, builder, sig.return_type, result)


@builtin
@implement('getitem', types.Kind(types.Buffer), types.slice3_type)
def getitem_array1d_slice(context, builder, sig, args):
    aryty, idxty = sig.args
    ary, idx = args

    ary = make_array(aryty)(context, builder, value=ary)

    res = _getitem_array_generic(context, builder, sig.return_type,
                                 aryty, ary, (idxty,), (idx,))
    return impl_ret_borrowed(context, builder, sig.return_type, res)


@builtin
@implement('getitem', types.Kind(types.Buffer), types.Kind(types.BaseTuple))
def getitem_array_tuple(context, builder, sig, args):
    aryty, tupty = sig.args
    ary, tup = args
    ary = make_array(aryty)(context, builder, ary)

    index_types = tupty.types
    indices = cgutils.unpack_tuple(builder, tup, count=len(tupty))

    if any(isinstance(ty, types.Array) for ty in index_types):
        return fancy_getitem(context, builder, sig, args,
                             aryty, ary, index_types, indices)

    dataptr, shapes, strides = \
        basic_indexing(context, builder, aryty, ary, index_types, indices)

    ndim = aryty.ndim
    if isinstance(sig.return_type, types.Array):
        # Generic array slicing
        res = make_view(context, builder, aryty, ary, sig.return_type,
                        dataptr, shapes, strides)
        res = res._getvalue()
    else:
        # Plain indexing (returning a scalar)
        assert not shapes
        res = load_item(context, builder, aryty, dataptr)

    return impl_ret_borrowed(context, builder ,sig.return_type, res)


@builtin
@implement('setitem', types.Kind(types.Buffer), types.Any, types.Any)
def setitem_array(context, builder, sig, args):
    """
    array[a] = scalar_or_array
    array[a,..,b] = scalar_or_array
    """
    aryty, idxty, valty = sig.args
    ary, idx, val = args

    if isinstance(idxty, types.BaseTuple):
        index_types = idxty.types
        indices = cgutils.unpack_tuple(builder, idx, count=len(idxty))
    else:
        index_types = (idxty,)
        indices = (idx,)

    ary = make_array(aryty)(context, builder, ary)

    # First try basic indexing to see if a single array location is denoted.
    try:
        dataptr, shapes, strides = \
            basic_indexing(context, builder, aryty, ary, index_types, indices)
    except NotImplementedError:
        use_fancy_indexing = True
    else:
        use_fancy_indexing = bool(shapes)

    if use_fancy_indexing:
        # Index describes a non-trivial view => use generic slice assignment
        # (NOTE: this also handles scalar broadcasting)
        return fancy_setslice(context, builder, sig, args,
                              index_types, indices)

    # Store source value the given location
    val = context.cast(builder, val, valty, aryty.dtype)
    store_item(context, builder, aryty, val, dataptr)


@builtin
@implement(types.len_type, types.Kind(types.Buffer))
def array_len(context, builder, sig, args):
    (aryty,) = sig.args
    (ary,) = args
    arystty = make_array(aryty)
    ary = arystty(context, builder, ary)
    shapeary = ary.shape
    res = builder.extract_value(shapeary, 0)
    return impl_ret_untracked(context, builder, sig.return_type, res)


#-------------------------------------------------------------------------------
# Advanced / fancy indexing


class Indexer(object):
    """
    Generic indexer interface, for generating indices over a fancy indexed
    array on a single dimension.
    """

    def prepare(self):
        """
        Prepare the indexer by initializing any required variables, basic
        blocks...
        """
        raise NotImplementedError

    def get_size(self):
        """
        Return this dimension's size as an integer.
        """
        raise NotImplementedError

    def get_shape(self):
        """
        Return this dimension's shape as a tuple.
        """
        raise NotImplementedError

    def loop_head(self):
        """
        Start indexation loop.  Return a (index, count) tuple.
        *index* is an integer LLVM value representing the index over this
        dimension.
        *count* is either an integer LLVM value representing the current
        iteration count, or None if this dimension should be omitted from
        the indexation result.
        """
        raise NotImplementedError

    def loop_tail(self):
        """
        Finish indexation loop.
        """
        raise NotImplementedError


class EntireIndexer(Indexer):
    """
    Compute indices along an entire array dimension.
    """

    def __init__(self, context, builder, aryty, ary, dim):
        self.context = context
        self.builder = builder
        self.aryty = aryty
        self.ary = ary
        self.dim = dim
        self.ll_intp = self.context.get_value_type(types.intp)

    def prepare(self):
        builder = self.builder
        self.size = builder.extract_value(self.ary.shape, self.dim)
        self.index = cgutils.alloca_once(builder, self.ll_intp)
        self.bb_start = builder.append_basic_block()
        self.bb_end = builder.append_basic_block()

    def get_size(self):
        return self.size

    def get_shape(self):
        return (self.size,)

    def loop_head(self):
        builder = self.builder
        # Initialize loop variable
        self.builder.store(Constant.int(self.ll_intp, 0), self.index)
        builder.branch(self.bb_start)
        builder.position_at_end(self.bb_start)
        cur_index = builder.load(self.index)
        with builder.if_then(builder.icmp_signed('>=', cur_index, self.size),
                             likely=False):
            builder.branch(self.bb_end)
        return cur_index, cur_index

    def loop_tail(self):
        builder = self.builder
        next_index = builder.add(builder.load(self.index),
                                 self.context.get_constant(types.intp, 1))
        builder.store(next_index, self.index)
        builder.branch(self.bb_start)
        builder.position_at_end(self.bb_end)


class IntegerIndexer(Indexer):
    """
    Compute indices from a single integer.
    """

    def __init__(self, context, builder, idx):
        self.context = context
        self.builder = builder
        self.idx = idx
        self.ll_intp = self.context.get_value_type(types.intp)

    def prepare(self):
        pass

    def get_size(self):
        return Constant.int(self.ll_intp, 1)

    def get_shape(self):
        return ()

    def loop_head(self):
        return self.idx, None

    def loop_tail(self):
        pass


class IntegerArrayIndexer(Indexer):
    """
    Compute indices from an array of integer indices.
    """

    def __init__(self, context, builder, idxty, idxary, size):
        self.context = context
        self.builder = builder
        self.idxty = idxty
        self.idxary = idxary
        self.size = size
        assert idxty.ndim == 1
        self.ll_intp = self.context.get_value_type(types.intp)

    def prepare(self):
        builder = self.builder
        self.idx_size = cgutils.unpack_tuple(builder, self.idxary.shape)[0]
        self.idx_index = cgutils.alloca_once(builder, self.ll_intp)
        self.bb_start = builder.append_basic_block()
        self.bb_end = builder.append_basic_block()

    def get_size(self):
        return self.idx_size

    def get_shape(self):
        return (self.idx_size,)

    def loop_head(self):
        builder = self.builder
        # Initialize loop variable
        self.builder.store(Constant.int(self.ll_intp, 0), self.idx_index)
        builder.branch(self.bb_start)
        builder.position_at_end(self.bb_start)
        cur_index = builder.load(self.idx_index)
        with builder.if_then(builder.icmp_signed('>=', cur_index, self.idx_size),
                             likely=False):
            builder.branch(self.bb_end)
        # Load the actual index from the array of indices
        index = _getitem_array1d(self.context, builder,
                                 self.idxty, self.idxary,
                                 cur_index, wraparound=False)
        index = fix_integer_index(self.context, builder,
                                  self.idxty.dtype, index, self.size)
        return index, cur_index

    def loop_tail(self):
        builder = self.builder
        next_index = builder.add(builder.load(self.idx_index),
                                 Constant.int(self.ll_intp, 1))
        builder.store(next_index, self.idx_index)
        builder.branch(self.bb_start)
        builder.position_at_end(self.bb_end)


class BooleanArrayIndexer(Indexer):
    """
    Compute indices from an array of boolean predicates.
    """

    def __init__(self, context, builder, idxty, idxary):
        self.context = context
        self.builder = builder
        self.idxty = idxty
        self.idxary = idxary
        assert idxty.ndim == 1
        self.ll_intp = self.context.get_value_type(types.intp)
        self.zero = Constant.int(self.ll_intp, 0)
        self.one = Constant.int(self.ll_intp, 1)

    def prepare(self):
        builder = self.builder
        self.size = cgutils.unpack_tuple(builder, self.idxary.shape)[0]
        self.idx_index = cgutils.alloca_once(builder, self.ll_intp)
        self.count = cgutils.alloca_once(builder, self.ll_intp)
        self.bb_start = builder.append_basic_block()
        self.bb_tail = builder.append_basic_block()
        self.bb_end = builder.append_basic_block()

    def get_size(self):
        builder = self.builder
        count = cgutils.alloca_once_value(builder, self.zero)
        # Sum all true values
        with cgutils.for_range(builder, self.size) as loop:
            c = builder.load(count)
            pred = _getitem_array1d(self.context, builder,
                                    self.idxty, self.idxary,
                                    loop.index, wraparound=False)
            c = builder.add(c, builder.zext(pred, c.type))
            builder.store(c, count)

        return builder.load(count)

    def get_shape(self):
        return (self.get_size(),)

    def loop_head(self):
        builder = self.builder
        # Initialize loop variable
        self.builder.store(self.zero, self.idx_index)
        self.builder.store(self.zero, self.count)
        builder.branch(self.bb_start)
        builder.position_at_end(self.bb_start)
        cur_index = builder.load(self.idx_index)
        cur_count = builder.load(self.count)
        with builder.if_then(builder.icmp_signed('>=', cur_index, self.size),
                             likely=False):
            builder.branch(self.bb_end)
        # Load the predicate and branch if false
        pred = _getitem_array1d(self.context, builder,
                                self.idxty, self.idxary,
                                cur_index, wraparound=False)
        with builder.if_then(builder.not_(pred)):
            builder.branch(self.bb_tail)
        # Increment the count for next iteration
        next_count = builder.add(cur_count, self.one)
        builder.store(next_count, self.count)
        return cur_index, cur_count

    def loop_tail(self):
        builder = self.builder
        builder.branch(self.bb_tail)
        builder.position_at_end(self.bb_tail)
        next_index = builder.add(builder.load(self.idx_index), self.one)
        builder.store(next_index, self.idx_index)
        builder.branch(self.bb_start)
        builder.position_at_end(self.bb_end)


class SliceIndexer(Indexer):
    """
    Compute indices along a slice.
    """

    def __init__(self, context, builder, aryty, ary, dim, slice):
        self.context = context
        self.builder = builder
        self.aryty = aryty
        self.ary = ary
        self.dim = dim
        self.slice = slice
        self.ll_intp = self.context.get_value_type(types.intp)
        self.zero = Constant.int(self.ll_intp, 0)
        self.one = Constant.int(self.ll_intp, 1)

    def prepare(self):
        builder = self.builder
        # Fix slice for the dimension's size
        self.dim_size = builder.extract_value(self.ary.shape, self.dim)
        cgutils.guard_invalid_slice(self.context, builder, self.slice)
        slicing.fix_slice(builder, self.slice, self.dim_size)
        self.is_step_negative = cgutils.is_neg_int(builder, self.slice.step)
        # Create loop entities
        self.index = cgutils.alloca_once(builder, self.ll_intp)
        self.count = cgutils.alloca_once(builder, self.ll_intp)
        self.bb_start = builder.append_basic_block()
        self.bb_end = builder.append_basic_block()

    def get_size(self):
        return slicing.get_slice_length(self.builder, self.slice)

    def get_shape(self):
        return (self.get_size(),)

    def loop_head(self):
        builder = self.builder
        # Initialize loop variable
        self.builder.store(self.slice.start, self.index)
        self.builder.store(self.zero, self.count)
        builder.branch(self.bb_start)
        builder.position_at_end(self.bb_start)
        cur_index = builder.load(self.index)
        cur_count = builder.load(self.count)
        is_finished = builder.select(self.is_step_negative,
                                     builder.icmp_signed('<=', cur_index,
                                                         self.slice.stop),
                                     builder.icmp_signed('>=', cur_index,
                                                         self.slice.stop))
        with builder.if_then(is_finished, likely=False):
            builder.branch(self.bb_end)
        return cur_index, cur_count

    def loop_tail(self):
        builder = self.builder
        next_index = builder.add(builder.load(self.index), self.slice.step)
        builder.store(next_index, self.index)
        next_count = builder.add(builder.load(self.count), self.one)
        builder.store(next_count, self.count)
        builder.branch(self.bb_start)
        builder.position_at_end(self.bb_end)


class FancyIndexer(object):
    """
    Perform fancy indexing on the given array.
    """

    def __init__(self, context, builder, aryty, ary, index_types, indices):
        self.context = context
        self.builder = builder
        self.aryty = ary
        self.shapes = cgutils.unpack_tuple(builder, ary.shape, aryty.ndim)
        self.strides = cgutils.unpack_tuple(builder, ary.strides, aryty.ndim)

        indexers = []

        ax = 0
        for indexval, idxty in zip(indices, index_types):
            if idxty is types.ellipsis:
                # Fill up missing dimensions at the middle
                n_missing = aryty.ndim - len(indices) + 1
                for i in range(n_missing):
                    indexer = EntireIndexer(context, builder, aryty, ary, ax)
                    indexers.append(indexer)
                    ax += 1
                continue

            # Regular index value
            if idxty == types.slice3_type:
                slice = slicing.Slice(context, builder, value=indexval)
                indexer = SliceIndexer(context, builder, aryty, ary, ax, slice)
                indexers.append(indexer)
            elif isinstance(idxty, types.Integer):
                ind = fix_integer_index(context, builder, idxty, indexval,
                                        self.shapes[ax])
                indexer = IntegerIndexer(context, builder, ind)
                indexers.append(indexer)
            elif isinstance(idxty, types.Array):
                idxary = make_array(idxty)(context, builder, indexval)
                if isinstance(idxty.dtype, types.Integer):
                    indexer = IntegerArrayIndexer(context, builder,
                                                  idxty, idxary,
                                                  self.shapes[ax])
                elif isinstance(idxty.dtype, types.Boolean):
                    indexer = BooleanArrayIndexer(context, builder,
                                                  idxty, idxary)
                else:
                    assert 0
                indexers.append(indexer)
            else:
                raise AssertionError("unexpected index type: %s" % (idxty,))
            ax += 1

        # Fill up missing dimensions at the end
        assert ax <= aryty.ndim, (ax, aryty.ndim)
        while ax < aryty.ndim:
            indexer = EntireIndexer(context, builder, aryty, ary, ax)
            indexers.append(indexer)
            ax += 1

        assert len(indexers) == aryty.ndim, (len(indexers), aryty.ndim)
        self.indexers = indexers

    def prepare(self):
        for i in self.indexers:
            i.prepare()

    def get_shape(self):
        """
        Get the resulting shape as Python tuple.
        """
        return sum([i.get_shape() for i in self.indexers], ())

    def begin_loops(self):
        indices, counts = zip(*(i.loop_head() for i in self.indexers))
        return indices, counts

    def end_loops(self):
        for i in reversed(self.indexers):
            i.loop_tail()


def fancy_getitem(context, builder, sig, args,
                  aryty, ary, index_types, indices):

    shapes = cgutils.unpack_tuple(builder, ary.shape)
    strides = cgutils.unpack_tuple(builder, ary.strides)
    data = ary.data

    indexer = FancyIndexer(context, builder, aryty, ary,
                           index_types, indices)
    indexer.prepare()

    # Construct output array
    out_ty = sig.return_type
    out_shapes = indexer.get_shape()

    out = _empty_nd_impl(context, builder, out_ty, out_shapes)
    out_data = out.data
    out_idx = cgutils.alloca_once_value(builder,
                                        context.get_constant(types.intp, 0))

    # Loop on source and copy to destination
    indices, _ = indexer.begin_loops()

    # No need to check for wraparound, as the indexers all ensure
    # a positive index is returned.
    ptr = cgutils.get_item_pointer2(builder, data, shapes, strides,
                                    aryty.layout, indices, wraparound=False)
    val = load_item(context, builder, aryty, ptr)

    # Since the destination is C-contiguous, no need for multi-dimensional
    # indexing.
    cur = builder.load(out_idx)
    ptr = builder.gep(out_data, [cur])
    store_item(context, builder, out_ty, val, ptr)
    next_idx = builder.add(cur, context.get_constant(types.intp, 1))
    builder.store(next_idx, out_idx)

    indexer.end_loops()

    return impl_ret_new_ref(context, builder, out_ty, out._getvalue())


@builtin
@implement('getitem', types.Kind(types.Buffer), types.Kind(types.Array))
def fancy_getitem_array(context, builder, sig, args):
    aryty, idxty = sig.args
    ary, idx = args
    ary = make_array(aryty)(context, builder, ary)
    out_ty = sig.return_type

    return fancy_getitem(context, builder, sig, args,
                         aryty, ary, (idxty,), (idx,))


def fancy_setslice(context, builder, sig, args, index_types, indices):
    """
    Implement slice assignment for arrays.  This implementation works for
    basic as well as fancy indexing, since there's no functional difference
    between the two for indexed assignment.
    """
    aryty, _, srcty = sig.args
    ary, _, src = args

    ary = make_array(aryty)(context, builder, ary)
    dest_shapes = cgutils.unpack_tuple(builder, ary.shape)
    dest_strides = cgutils.unpack_tuple(builder, ary.strides)
    dest_data = ary.data

    indexer = FancyIndexer(context, builder, aryty, ary,
                           index_types, indices)
    indexer.prepare()

    if isinstance(srcty, types.Buffer):
        # Source is an array
        src = make_array(srcty)(context, builder, src)
        src_shapes = cgutils.unpack_tuple(builder, src.shape)
        src_strides = cgutils.unpack_tuple(builder, src.strides)
        src_data = src.data
        src_dtype = srcty.dtype

        # Check shapes are equal
        index_shape = indexer.get_shape()
        shape_error = cgutils.false_bit
        assert len(index_shape) == len(src_shapes)

        for u, v in zip(src_shapes, index_shape):
            shape_error = builder.or_(shape_error,
                                      builder.icmp_signed('!=', u, v))

        with builder.if_then(shape_error, likely=False):
            msg = "cannot assign slice from input of different size"
            context.call_conv.return_user_exc(builder, ValueError, (msg,))

        def src_getitem(source_indices):
            assert len(source_indices) == srcty.ndim
            src_ptr = cgutils.get_item_pointer2(builder, src_data,
                                                src_shapes, src_strides,
                                                srcty.layout, source_indices,
                                                wraparound=False)
            return load_item(context, builder, srcty, src_ptr)

    else:
        # Source is a scalar (broadcast or not, depending on destination
        # shape).
        src_dtype = srcty

        def src_getitem(source_indices):
            return src

    # Loop on destination and copy from source to destination
    dest_indices, counts = indexer.begin_loops()

    # Source is iterated in natural order
    source_indices = tuple(c for c in counts if c is not None)
    val = src_getitem(source_indices)

    # Cast to the destination dtype (cross-dtype slice assignement is allowed)
    val = context.cast(builder, val, src_dtype, aryty.dtype)

    # No need to check for wraparound, as the indexers all ensure
    # a positive index is returned.
    dest_ptr = cgutils.get_item_pointer2(builder, dest_data,
                                         dest_shapes, dest_strides,
                                         aryty.layout, dest_indices,
                                         wraparound=False)
    store_item(context, builder, aryty, val, dest_ptr)

    indexer.end_loops()

    return context.get_dummy_value()


#-------------------------------------------------------------------------------
# Shape / layout altering

@builtin
@implement('array.transpose', types.Kind(types.Array))
def array_transpose(context, builder, sig, args):
    return array_T(context, builder, sig.args[0], args[0])

def array_T(context, builder, typ, value):
    if typ.ndim <= 1:
        res = value
    else:
        ary = make_array(typ)(context, builder, value)
        ret = make_array(typ)(context, builder)
        shapes = cgutils.unpack_tuple(builder, ary.shape, typ.ndim)
        strides = cgutils.unpack_tuple(builder, ary.strides, typ.ndim)
        populate_array(ret,
                       data=ary.data,
                       shape=cgutils.pack_array(builder, shapes[::-1]),
                       strides=cgutils.pack_array(builder, strides[::-1]),
                       itemsize=ary.itemsize,
                       meminfo=ary.meminfo,
                       parent=ary.parent)
        res = ret._getvalue()
    return impl_ret_borrowed(context, builder, typ, res)

builtin_attr(impl_attribute(types.Kind(types.Array), 'T')(array_T))


def _attempt_nocopy_reshape(context, builder, aryty, ary, newnd, newshape,
                            newstrides):
    """
    Call into Numba_attempt_nocopy_reshape() for the given array type
    and instance, and the specified new shape.  The array pointed to
    by *newstrides* will be filled up if successful.
    """
    ll_intp = context.get_value_type(types.intp)
    ll_intp_star = ll_intp.as_pointer()
    ll_intc = context.get_value_type(types.intc)
    fnty = lc.Type.function(ll_intc, [ll_intp, ll_intp_star, ll_intp_star,
                                      ll_intp, ll_intp_star, ll_intp_star,
                                      ll_intp, ll_intc])
    fn = builder.module.get_or_insert_function(
        fnty, name="numba_attempt_nocopy_reshape")

    nd = lc.Constant.int(ll_intp, aryty.ndim)
    shape = cgutils.gep_inbounds(builder, ary._get_ptr_by_name('shape'), 0, 0)
    strides = cgutils.gep_inbounds(builder, ary._get_ptr_by_name('strides'), 0, 0)
    newnd = lc.Constant.int(ll_intp, newnd)
    newshape = cgutils.gep_inbounds(builder, newshape, 0, 0)
    newstrides = cgutils.gep_inbounds(builder, newstrides, 0, 0)
    is_f_order = lc.Constant.int(ll_intc, 0)
    res = builder.call(fn, [nd, shape, strides,
                            newnd, newshape, newstrides,
                            ary.itemsize, is_f_order])
    return res

@builtin
@implement('array.reshape', types.Kind(types.Array), types.Kind(types.BaseTuple))
def array_reshape(context, builder, sig, args):
    aryty = sig.args[0]
    retty = sig.return_type
    shapety = sig.args[1]
    shape = args[1]

    ll_intp = context.get_value_type(types.intp)
    ll_shape = lc.Type.array(ll_intp, shapety.count)

    ary = make_array(aryty)(context, builder, args[0])

    # XXX unknown dimension (-1) is unhandled
    msg = "negative shape is not handled, yet"
    for s in cgutils.unpack_tuple(builder, shape):
        is_neg = builder.icmp_signed('<', s, lc.Constant.int(ll_intp, 0))
        with cgutils.if_unlikely(builder, is_neg):
            context.call_conv.return_user_exc(builder, NotImplementedError,
                                              (msg,))

    # Check requested size
    newsize = lc.Constant.int(ll_intp, 1)
    for s in cgutils.unpack_tuple(builder, shape):
        newsize = builder.mul(newsize, s)
    size = lc.Constant.int(ll_intp, 1)
    for s in cgutils.unpack_tuple(builder, ary.shape):
        size = builder.mul(size, s)
    fail = builder.icmp_unsigned('!=', size, newsize)
    with builder.if_then(fail):
        msg = "total size of new array must be unchanged"
        context.call_conv.return_user_exc(builder, ValueError, (msg,))

    newnd = shapety.count
    newshape = cgutils.alloca_once(builder, ll_shape)
    builder.store(shape, newshape)
    newstrides = cgutils.alloca_once(builder, ll_shape)

    ok = _attempt_nocopy_reshape(context, builder, aryty, ary, newnd,
                                 newshape, newstrides)
    fail = builder.icmp_unsigned('==', ok, lc.Constant.int(ok.type, 0))

    with builder.if_then(fail):
        msg = "incompatible shape for array"
        context.call_conv.return_user_exc(builder, NotImplementedError, (msg,))

    ret = make_array(retty)(context, builder)
    populate_array(ret,
                   data=ary.data,
                   shape=builder.load(newshape),
                   strides=builder.load(newstrides),
                   itemsize=ary.itemsize,
                   meminfo=ary.meminfo,
                   parent=ary.parent)
    res = ret._getvalue()
    return impl_ret_borrowed(context, builder, sig.return_type, res)

@builtin
@implement('array.reshape', types.Kind(types.Array), types.VarArg(types.Any))
def array_reshape_vararg(context, builder, sig, args):
    # types
    aryty = sig.args[0]
    dimtys = sig.args[1:]
    # values
    ary = args[0]
    dims = args[1:]
    # coerce all types to intp
    dims = [context.cast(builder, val, ty, types.intp)
            for ty, val in zip(dimtys, dims)]
    # make a tuple
    shape = cgutils.pack_array(builder, dims, dims[0].type)

    shapety = types.UniTuple(dtype=types.intp, count=len(dims))
    new_sig = typing.signature(sig.return_type, aryty, shapety)
    new_args = ary, shape
    return array_reshape(context, builder, new_sig, new_args)


def _change_dtype(context, builder, oldty, newty, ary):
    """
    Attempt to fix up *ary* for switching from *oldty* to *newty*.

    See Numpy's array_descr_set()
    (np/core/src/multiarray/getset.c).
    Attempt to fix the array's shape and strides for a new dtype.
    False is returned on failure, True on success.
    """
    assert oldty.ndim == newty.ndim
    assert oldty.layout == newty.layout

    new_layout = ord(newty.layout)
    any_layout = ord('A')
    c_layout = ord('C')
    f_layout = ord('F')

    int8 = types.int8

    def imp(nd, dims, strides, old_itemsize, new_itemsize, layout):
        # Attempt to update the layout due to limitation of the numba
        # type system.
        if layout == any_layout:
            # Test rightmost stride to be contiguous
            if strides[-1] == old_itemsize:
                # Process this as if it is C contiguous
                layout = int8(c_layout)
            # Test leftmost stride to be F contiguous
            elif strides[0] == old_itemsize:
                # Process this as if it is F contiguous
                layout = int8(f_layout)

        if old_itemsize != new_itemsize and (layout == any_layout or nd == 0):
            return False

        if layout == c_layout:
            i = nd - 1
        else:
            i = 0

        if new_itemsize < old_itemsize:
            # If it is compatible, increase the size of the dimension
            # at the end (or at the front if F-contiguous)
            if (old_itemsize % new_itemsize) != 0:
                return False

            newdim = old_itemsize // new_itemsize
            dims[i] *= newdim
            strides[i] = new_itemsize

        elif new_itemsize > old_itemsize:
            # Determine if last (or first if F-contiguous) dimension
            # is compatible
            bytelength = dims[i] * old_itemsize
            if (bytelength % new_itemsize) != 0:
                return False

            dims[i] = bytelength // new_itemsize
            strides[i] = new_itemsize

        else:
            # Same item size: nothing to do (this also works for
            # non-contiguous arrays).
            pass

        return True

    old_itemsize = context.get_constant(types.intp,
                                        get_itemsize(context, oldty))
    new_itemsize = context.get_constant(types.intp,
                                        get_itemsize(context, newty))

    nd = context.get_constant(types.intp, newty.ndim)
    shape_data = cgutils.gep_inbounds(builder, ary._get_ptr_by_name('shape'),
                                      0, 0)
    strides_data = cgutils.gep_inbounds(builder,
                                        ary._get_ptr_by_name('strides'), 0, 0)

    shape_strides_array_type = types.Array(dtype=types.intp, ndim=1, layout='C')
    arycls = context.make_array(shape_strides_array_type)

    shape_constant = cgutils.pack_array(builder,
                                        [context.get_constant(types.intp,
                                                              newty.ndim)])

    sizeof_intp = context.get_abi_sizeof(context.get_data_type(types.intp))
    sizeof_intp = context.get_constant(types.intp, sizeof_intp)
    strides_constant = cgutils.pack_array(builder, [sizeof_intp])

    shape_ary = arycls(context, builder)

    populate_array(shape_ary,
                   data=shape_data,
                   shape=shape_constant,
                   strides=strides_constant,
                   itemsize=sizeof_intp,
                   meminfo=None)

    strides_ary = arycls(context, builder)
    populate_array(strides_ary,
                   data=strides_data,
                   shape=shape_constant,
                   strides=strides_constant,
                   itemsize=sizeof_intp,
                   meminfo=None)

    shape = shape_ary._getvalue()
    strides = strides_ary._getvalue()
    args = [nd, shape, strides, old_itemsize, new_itemsize,
            context.get_constant(types.int8, new_layout)]

    sig = signature(types.boolean,
                    types.intp,  # nd
                    shape_strides_array_type,  # dims
                    shape_strides_array_type,  # strides
                    types.intp,  # old_itemsize
                    types.intp,  # new_itemsize
                    types.int8,  # layout
                    )

    res = context.compile_internal(builder, imp, sig, args)
    update_array_info(newty, ary)
    res = impl_ret_borrowed(context, builder, sig.return_type, res)
    return res


@builtin
@implement('array.view', types.Kind(types.Array), types.Kind(types.DTypeSpec))
def array_view(context, builder, sig, args):
    aryty = sig.args[0]
    retty = sig.return_type

    ary = make_array(aryty)(context, builder, args[0])
    ret = make_array(retty)(context, builder)
    # Copy all fields, casting the "data" pointer appropriately
    fields = set(ret._datamodel._fields)
    for k in sorted(fields):
        val = getattr(ary, k)
        if k == 'data':
            ptrty = ret.data.type
            ret.data = builder.bitcast(val, ptrty)
        else:
            setattr(ret, k, val)

    ok = _change_dtype(context, builder, aryty, retty, ret)
    fail = builder.icmp_unsigned('==', ok, lc.Constant.int(ok.type, 0))

    with builder.if_then(fail):
        msg = "new type not compatible with array"
        context.call_conv.return_user_exc(builder, ValueError, (msg,))

    res = ret._getvalue()
    return impl_ret_borrowed(context, builder, sig.return_type, res)


#-------------------------------------------------------------------------------
# Array attributes

@builtin_attr
@impl_attribute(types.Kind(types.Array), "dtype", types.Kind(types.DType))
def array_dtype(context, builder, typ, value):
    res = context.get_dummy_value()
    return impl_ret_untracked(context, builder, typ, res)

@builtin_attr
@impl_attribute(types.Kind(types.Array), "shape", types.Kind(types.UniTuple))
@impl_attribute(types.Kind(types.MemoryView), "shape", types.Kind(types.UniTuple))
def array_shape(context, builder, typ, value):
    arrayty = make_array(typ)
    array = arrayty(context, builder, value)
    res = array.shape
    return impl_ret_untracked(context, builder, typ, res)


@builtin_attr
@impl_attribute(types.Kind(types.Array), "strides", types.Kind(types.UniTuple))
@impl_attribute(types.Kind(types.MemoryView), "strides", types.Kind(types.UniTuple))
def array_strides(context, builder, typ, value):
    arrayty = make_array(typ)
    array = arrayty(context, builder, value)
    res = array.strides
    return impl_ret_untracked(context, builder, typ, res)


@builtin_attr
@impl_attribute(types.Kind(types.Array), "ndim", types.intp)
@impl_attribute(types.Kind(types.MemoryView), "ndim", types.intp)
def array_ndim(context, builder, typ, value):
    res = context.get_constant(types.intp, typ.ndim)
    return impl_ret_untracked(context, builder, typ, res)


@builtin_attr
@impl_attribute(types.Kind(types.Array), "size", types.intp)
def array_size(context, builder, typ, value):
    arrayty = make_array(typ)
    array = arrayty(context, builder, value)
    res = array.nitems
    return impl_ret_untracked(context, builder, typ, res)


@builtin_attr
@impl_attribute(types.Kind(types.Array), "itemsize", types.intp)
@impl_attribute(types.Kind(types.MemoryView), "itemsize", types.intp)
def array_itemsize(context, builder, typ, value):
    arrayty = make_array(typ)
    array = arrayty(context, builder, value)
    res = array.itemsize
    return impl_ret_untracked(context, builder, typ, res)


@builtin_attr
@impl_attribute(types.Kind(types.MemoryView), "nbytes", types.intp)
def array_nbytes(context, builder, typ, value):
    """
    nbytes = size * itemsize
    """
    arrayty = make_array(typ)
    array = arrayty(context, builder, value)
    dims = cgutils.unpack_tuple(builder, array.shape, typ.ndim)
    res = builder.mul(array.nitems, array.itemsize)
    return impl_ret_untracked(context, builder, typ, res)


@builtin_attr
@impl_attribute(types.Kind(types.MemoryView), "contiguous", types.boolean)
def array_contiguous(context, builder, typ, value):
    res = context.get_constant(types.boolean, typ.is_contig)
    return impl_ret_untracked(context, builder, typ, res)

@builtin_attr
@impl_attribute(types.Kind(types.MemoryView), "c_contiguous", types.boolean)
def array_c_contiguous(context, builder, typ, value):
    res = context.get_constant(types.boolean, typ.is_c_contig)
    return impl_ret_untracked(context, builder, typ, res)

@builtin_attr
@impl_attribute(types.Kind(types.MemoryView), "f_contiguous", types.boolean)
def array_f_contiguous(context, builder, typ, value):
    res = context.get_constant(types.boolean, typ.is_f_contig)
    return impl_ret_untracked(context, builder, typ, res)


@builtin_attr
@impl_attribute(types.Kind(types.MemoryView), "readonly", types.boolean)
def array_readonly(context, builder, typ, value):
    res = context.get_constant(types.boolean, not typ.mutable)
    return impl_ret_untracked(context, builder, typ, res)


@builtin_attr
@impl_attribute(types.Kind(types.Array), "ctypes",
                types.Kind(types.ArrayCTypes))
def array_ctypes(context, builder, typ, value):
    arrayty = make_array(typ)
    array = arrayty(context, builder, value)
    # Cast void* data to uintp
    addr = builder.ptrtoint(array.data, context.get_value_type(types.uintp))
    # Create new ArrayCType structure
    ctinfo_type = cgutils.create_struct_proxy(types.ArrayCTypes(typ))
    ctinfo = ctinfo_type(context, builder)
    ctinfo.data = addr
    res = ctinfo._getvalue()
    return impl_ret_untracked(context, builder, typ, res)


@builtin_attr
@impl_attribute(types.Kind(types.Array), "flags", types.Kind(types.ArrayFlags))
def array_flags(context, builder, typ, value):
    res = context.get_dummy_value()
    return impl_ret_untracked(context, builder, typ, res)


@builtin_attr
@impl_attribute(types.Kind(types.ArrayCTypes), "data", types.uintp)
def array_ctypes_data(context, builder, typ, value):
    ctinfo_type = cgutils.create_struct_proxy(typ)
    ctinfo = ctinfo_type(context, builder, value=value)
    res = ctinfo.data
    return impl_ret_untracked(context, builder, typ, res)


@builtin_attr
@impl_attribute(types.Kind(types.ArrayFlags), "contiguous", types.boolean)
@impl_attribute(types.Kind(types.ArrayFlags), "c_contiguous", types.boolean)
def array_ctypes_data(context, builder, typ, value):
    val = typ.array_type.layout == 'C'
    res = context.get_constant(types.boolean, val)
    return impl_ret_untracked(context, builder, typ, res)

@builtin_attr
@impl_attribute(types.Kind(types.ArrayFlags), "f_contiguous", types.boolean)
def array_ctypes_data(context, builder, typ, value):
    layout = typ.array_type.layout
    val = layout == 'F' if typ.array_type.ndim > 1 else layout in 'CF'
    res = context.get_constant(types.boolean, val)
    return impl_ret_untracked(context, builder, typ, res)


#-------------------------------------------------------------------------------
# Structured / record lookup

@builtin_attr
@impl_attribute_generic(types.Kind(types.Array))
def array_record_getattr(context, builder, typ, value, attr):
    """
    Generic getattr() implementation for record arrays: fetch the given
    record member, i.e. a subarray.
    """
    arrayty = make_array(typ)
    array = arrayty(context, builder, value)

    rectype = typ.dtype
    if not isinstance(rectype, types.Record):
        raise NotImplementedError("attribute %r of %s not defined" % (attr, typ))
    dtype = rectype.typeof(attr)
    offset = rectype.offset(attr)

    resty = typ.copy(dtype=dtype, layout='A')

    raryty = make_array(resty)

    rary = raryty(context, builder)

    constoffset = context.get_constant(types.intp, offset)

    llintp = context.get_value_type(types.intp)
    newdata = builder.add(builder.ptrtoint(array.data, llintp), constoffset)
    newdataptr = builder.inttoptr(newdata, rary.data.type)

    datasize = context.get_abi_sizeof(context.get_data_type(dtype))
    populate_array(rary,
                   data=newdataptr,
                   shape=array.shape,
                   strides=array.strides,
                   itemsize=context.get_constant(types.intp, datasize),
                   meminfo=array.meminfo,
                   parent=array.parent)
    res = rary._getvalue()
    return impl_ret_borrowed(context, builder, typ, res)

@builtin
@implement('static_getitem', types.Kind(types.Array), types.Kind(types.Const))
def array_record_getitem(context, builder, sig, args):
    return array_record_getattr(context, builder, sig.args[0], args[0], args[1])


@builtin_attr
@impl_attribute_generic(types.Kind(types.Record))
def record_getattr(context, builder, typ, value, attr):
    """
    Generic getattr() implementation for records: fetch the given
    record member, i.e. a scalar.
    """
    context.sentry_record_alignment(typ, attr)
    offset = typ.offset(attr)
    elemty = typ.typeof(attr)

    if isinstance(elemty, types.NestedArray):
        # Only a nested array's *data* is stored in a structured array,
        # so we create an array structure to point to that data.
        aryty = make_array(elemty)
        ary = aryty(context, builder)
        dtype = elemty.dtype
        newshape = [context.get_constant(types.intp, s) for s in
                    elemty.shape]
        newstrides = [context.get_constant(types.intp, s) for s in
                      elemty.strides]
        newdata = cgutils.get_record_member(builder, value, offset,
                                            context.get_data_type(dtype))
        populate_array(
            ary,
            data=newdata,
            shape=cgutils.pack_array(builder, newshape),
            strides=cgutils.pack_array(builder, newstrides),
            itemsize=context.get_constant(types.intp, elemty.size),
            meminfo=None,
            parent=None,
        )
        res = ary._getvalue()
        return impl_ret_borrowed(context, builder, typ, res)
    else:
        dptr = cgutils.get_record_member(builder, value, offset,
                                         context.get_data_type(elemty))
        align = None if typ.aligned else 1
        res = context.unpack_value(builder, elemty, dptr, align)
        return impl_ret_borrowed(context, builder, typ, res)

@builtin
@implement('static_getitem', types.Kind(types.Record), types.Kind(types.Const))
def record_getitem(context, builder, sig, args):
    """
    Record.__getitem__ redirects to getattr()
    """
    impl = context.get_attribute(args[0], sig.args[0], args[1])
    return impl(context, builder, sig.args[0], args[0], args[1])

@builtin
@implement('static_setitem', types.Kind(types.Record), types.Kind(types.Const), types.Any)
def record_setitem(context, builder, sig, args):
    """
    Record.__setitem__ redirects to setattr()
    """
    recty, _, valty = sig.args
    rec, idx, val = args
    getattr_sig = signature(sig.return_type, recty, valty)
    impl = context.get_setattr(idx, getattr_sig)
    assert impl is not None
    return impl(builder, (rec, val))


#-------------------------------------------------------------------------------
# Comparisons

@builtin
@implement('is', types.Kind(types.Array), types.Kind(types.Array))
def array_is(context, builder, sig, args):
    aty, bty = sig.args
    if aty != bty:
        return cgutils.false_bit

    def array_is_impl(a, b):
        return (a.shape == b.shape and
                a.strides == b.strides and
                a.ctypes.data == b.ctypes.data)

    return context.compile_internal(builder, array_is_impl, sig, args)


#-------------------------------------------------------------------------------
# builtin `numpy.flat` implementation

def make_array_flat_cls(flatiterty):
    """
    Return the Structure representation of the given *flatiterty* (an
    instance of types.NumpyFlatType).
    """
    return _make_flattening_iter_cls(flatiterty, 'flat')


def make_array_ndenumerate_cls(nditerty):
    """
    Return the Structure representation of the given *nditerty* (an
    instance of types.NumpyNdEnumerateType).
    """
    return _make_flattening_iter_cls(nditerty, 'ndenumerate')


def _increment_indices(context, builder, ndim, shape, indices, end_flag=None):
    zero = context.get_constant(types.intp, 0)
    one = context.get_constant(types.intp, 1)

    bbend = builder.append_basic_block('end_increment')

    if end_flag is not None:
        builder.store(cgutils.false_byte, end_flag)

    for dim in reversed(range(ndim)):
        idxptr = cgutils.gep_inbounds(builder, indices, dim)
        idx = increment_index(builder, builder.load(idxptr))

        count = shape[dim]
        in_bounds = builder.icmp(lc.ICMP_SLT, idx, count)
        with cgutils.if_likely(builder, in_bounds):
            builder.store(idx, idxptr)
            builder.branch(bbend)
        builder.store(zero, idxptr)

    if end_flag is not None:
        builder.store(cgutils.true_byte, end_flag)
    builder.branch(bbend)

    builder.position_at_end(bbend)

def _increment_indices_array(context, builder, arrty, arr, indices, end_flag=None):
    shape = cgutils.unpack_tuple(builder, arr.shape, arrty.ndim)
    _increment_indices(context, builder, arrty.ndim, shape, indices, end_flag)


def make_ndindex_cls(nditerty):
    """
    Return the Structure representation of the given *nditerty* (an
    instance of types.NumpyNdIndexType).
    """
    ndim = nditerty.ndim

    class NdIndexIter(cgutils.create_struct_proxy(nditerty)):
        """
        .ndindex() implementation.
        """

        def init_specific(self, context, builder, shapes):
            zero = context.get_constant(types.intp, 0)
            indices = cgutils.alloca_once(builder, zero.type,
                                          size=context.get_constant(types.intp,
                                                                    ndim))
            exhausted = cgutils.alloca_once_value(builder, cgutils.false_byte)

            for dim in range(ndim):
                idxptr = cgutils.gep_inbounds(builder, indices, dim)
                builder.store(zero, idxptr)
                # 0-sized dimensions really indicate an empty array,
                # but we have to catch that condition early to avoid
                # a bug inside the iteration logic.
                dim_size = shapes[dim]
                dim_is_empty = builder.icmp(lc.ICMP_EQ, dim_size, zero)
                with cgutils.if_unlikely(builder, dim_is_empty):
                    builder.store(cgutils.true_byte, exhausted)

            self.indices = indices
            self.exhausted = exhausted
            self.shape = cgutils.pack_array(builder, shapes, zero.type)

        def iternext_specific(self, context, builder, result):
            zero = context.get_constant(types.intp, 0)
            one = context.get_constant(types.intp, 1)

            bbend = builder.append_basic_block('end')

            exhausted = cgutils.as_bool_bit(builder, builder.load(self.exhausted))
            with cgutils.if_unlikely(builder, exhausted):
                result.set_valid(False)
                builder.branch(bbend)

            indices = [builder.load(cgutils.gep_inbounds(builder, self.indices, dim))
                       for dim in range(ndim)]
            for load in indices:
                mark_positive(builder, load)

            result.yield_(cgutils.pack_array(builder, indices, zero.type))
            result.set_valid(True)

            shape = cgutils.unpack_tuple(builder, self.shape, ndim)
            _increment_indices(context, builder, ndim, shape,
                               self.indices, self.exhausted)

            builder.branch(bbend)
            builder.position_at_end(bbend)

    return NdIndexIter


def _make_flattening_iter_cls(flatiterty, kind):
    assert kind in ('flat', 'ndenumerate')

    array_type = flatiterty.array_type
    dtype = array_type.dtype

    if array_type.layout == 'C':
        class CContiguousFlatIter(cgutils.create_struct_proxy(flatiterty)):
            """
            .flat() / .ndenumerate() implementation for C-contiguous arrays.
            """

            def init_specific(self, context, builder, arrty, arr):
                zero = context.get_constant(types.intp, 0)
                self.index = cgutils.alloca_once_value(builder, zero)
                # We can't trust strides[-1] to always contain the right
                # step value, see
                # http://docs.scipy.org/doc/numpy-dev/release.html#npy-relaxed-strides-checking
                self.stride = arr.itemsize

                if kind == 'ndenumerate':
                    # Zero-initialize the indices array.
                    indices = cgutils.alloca_once(
                        builder, zero.type,
                        size=context.get_constant(types.intp, arrty.ndim))

                    for dim in range(arrty.ndim):
                        idxptr = cgutils.gep_inbounds(builder, indices, dim)
                        builder.store(zero, idxptr)

                    self.indices = indices

            # NOTE: Using gep() instead of explicit pointer addition helps
            # LLVM vectorize the loop (since the stride is known and
            # constant).  This is not possible in the non-contiguous case,
            # where the strides are unknown at compile-time.

            def iternext_specific(self, context, builder, arrty, arr, result):
                zero = context.get_constant(types.intp, 0)
                one = context.get_constant(types.intp, 1)

                ndim = arrty.ndim
                nitems = arr.nitems

                index = builder.load(self.index)
                is_valid = builder.icmp(lc.ICMP_SLT, index, nitems)
                result.set_valid(is_valid)

                with cgutils.if_likely(builder, is_valid):
                    ptr = builder.gep(arr.data, [index])
                    value = load_item(context, builder, arrty, ptr)
                    if kind == 'flat':
                        result.yield_(value)
                    else:
                        # ndenumerate(): fetch and increment indices
                        indices = self.indices
                        idxvals = [builder.load(cgutils.gep_inbounds(builder, indices, dim))
                                   for dim in range(ndim)]
                        idxtuple = cgutils.pack_array(builder, idxvals)
                        result.yield_(
                            cgutils.make_anonymous_struct(builder, [idxtuple, value]))
                        _increment_indices_array(context, builder, arrty, arr, indices)

                    index = builder.add(index, one)
                    builder.store(index, self.index)

            def getitem(self, context, builder, arrty, arr, index):
                ptr = builder.gep(arr.data, [index])
                return load_item(context, builder, arrty, ptr)

            def setitem(self, context, builder, arrty, arr, index, value):
                ptr = builder.gep(arr.data, [index])
                store_item(context, builder, arrty, value, ptr)

        return CContiguousFlatIter

    else:
        class FlatIter(cgutils.create_struct_proxy(flatiterty)):
            """
            Generic .flat() / .ndenumerate() implementation for
            non-contiguous arrays.
            It keeps track of pointers along each dimension in order to
            minimize computations.
            """

            def init_specific(self, context, builder, arrty, arr):
                zero = context.get_constant(types.intp, 0)
                data = arr.data
                ndim = arrty.ndim
                shapes = cgutils.unpack_tuple(builder, arr.shape, ndim)

                indices = cgutils.alloca_once(builder, zero.type,
                                              size=context.get_constant(types.intp,
                                                                        arrty.ndim))
                pointers = cgutils.alloca_once(builder, data.type,
                                               size=context.get_constant(types.intp,
                                                                         arrty.ndim))
                strides = cgutils.unpack_tuple(builder, arr.strides, ndim)
                exhausted = cgutils.alloca_once_value(builder, cgutils.false_byte)

                # Initialize indices and pointers with their start values.
                for dim in range(ndim):
                    idxptr = cgutils.gep_inbounds(builder, indices, dim)
                    ptrptr = cgutils.gep_inbounds(builder, pointers, dim)
                    builder.store(data, ptrptr)
                    builder.store(zero, idxptr)
                    # 0-sized dimensions really indicate an empty array,
                    # but we have to catch that condition early to avoid
                    # a bug inside the iteration logic (see issue #846).
                    dim_size = shapes[dim]
                    dim_is_empty = builder.icmp(lc.ICMP_EQ, dim_size, zero)
                    with cgutils.if_unlikely(builder, dim_is_empty):
                        builder.store(cgutils.true_byte, exhausted)

                self.indices = indices
                self.pointers = pointers
                self.exhausted = exhausted

            def iternext_specific(self, context, builder, arrty, arr, result):
                ndim = arrty.ndim
                data = arr.data
                shapes = cgutils.unpack_tuple(builder, arr.shape, ndim)
                strides = cgutils.unpack_tuple(builder, arr.strides, ndim)
                indices = self.indices
                pointers = self.pointers

                zero = context.get_constant(types.intp, 0)
                one = context.get_constant(types.intp, 1)

                bbend = builder.append_basic_block('end')

                # Catch already computed iterator exhaustion
                is_exhausted = cgutils.as_bool_bit(
                    builder, builder.load(self.exhausted))
                with cgutils.if_unlikely(builder, is_exhausted):
                    result.set_valid(False)
                    builder.branch(bbend)
                result.set_valid(True)

                # Current pointer inside last dimension
                last_ptr = cgutils.gep_inbounds(builder, pointers, ndim - 1)
                ptr = builder.load(last_ptr)
                value = load_item(context, builder, arrty, ptr)
                if kind == 'flat':
                    result.yield_(value)
                else:
                    # ndenumerate() => yield (indices, value)
                    idxvals = [builder.load(cgutils.gep_inbounds(builder, indices, dim))
                               for dim in range(ndim)]
                    idxtuple = cgutils.pack_array(builder, idxvals)
                    result.yield_(
                        cgutils.make_anonymous_struct(builder, [idxtuple, value]))

                # Update indices and pointers by walking from inner
                # dimension to outer.
                for dim in reversed(range(ndim)):
                    idxptr = cgutils.gep_inbounds(builder, indices, dim)
                    idx = builder.add(builder.load(idxptr), one)

                    count = shapes[dim]
                    stride = strides[dim]
                    in_bounds = builder.icmp(lc.ICMP_SLT, idx, count)
                    with cgutils.if_likely(builder, in_bounds):
                        # Index is valid => pointer can simply be incremented.
                        builder.store(idx, idxptr)
                        ptrptr = cgutils.gep_inbounds(builder, pointers, dim)
                        ptr = builder.load(ptrptr)
                        ptr = cgutils.pointer_add(builder, ptr, stride)
                        builder.store(ptr, ptrptr)
                        # Reset pointers in inner dimensions
                        for inner_dim in range(dim + 1, ndim):
                            ptrptr = cgutils.gep_inbounds(builder, pointers, inner_dim)
                            builder.store(ptr, ptrptr)
                        builder.branch(bbend)
                    # Reset index and continue with next dimension
                    builder.store(zero, idxptr)

                # End of array
                builder.store(cgutils.true_byte, self.exhausted)
                builder.branch(bbend)

                builder.position_at_end(bbend)

            def _ptr_for_index(self, context, builder, arrty, arr, index):
                ndim = arrty.ndim
                shapes = cgutils.unpack_tuple(builder, arr.shape, count=ndim)
                strides = cgutils.unpack_tuple(builder, arr.strides, count=ndim)

                # First convert the flattened index into a regular n-dim index
                indices = []
                for dim in reversed(range(ndim)):
                    indices.append(builder.urem(index, shapes[dim]))
                    index = builder.udiv(index, shapes[dim])
                indices.reverse()

                ptr = cgutils.get_item_pointer2(builder, arr.data, shapes,
                                                strides, arrty.layout, indices)
                return ptr

            def getitem(self, context, builder, arrty, arr, index):
                ptr = self._ptr_for_index(context, builder, arrty, arr, index)
                return load_item(context, builder, arrty, ptr)

            def setitem(self, context, builder, arrty, arr, index, value):
                ptr = self._ptr_for_index(context, builder, arrty, arr, index)
                store_item(context, builder, arrty, value, ptr)

        return FlatIter


@builtin_attr
@impl_attribute(types.Kind(types.Array), "flat", types.Kind(types.NumpyFlatType))
def make_array_flatiter(context, builder, arrty, arr):
    flatitercls = make_array_flat_cls(types.NumpyFlatType(arrty))
    flatiter = flatitercls(context, builder)

    arrayptr = cgutils.alloca_once_value(builder, arr)
    flatiter.array = arrayptr

    arrcls = context.make_array(arrty)
    arr = arrcls(context, builder, ref=arrayptr)

    flatiter.init_specific(context, builder, arrty, arr)

    res = flatiter._getvalue()
    return impl_ret_borrowed(context, builder, types.NumpyFlatType(arrty), res)


@builtin
@implement('iternext', types.Kind(types.NumpyFlatType))
@iternext_impl
def iternext_numpy_flatiter(context, builder, sig, args, result):
    [flatiterty] = sig.args
    [flatiter] = args

    flatitercls = make_array_flat_cls(flatiterty)
    flatiter = flatitercls(context, builder, value=flatiter)

    arrty = flatiterty.array_type
    arrcls = context.make_array(arrty)
    arr = arrcls(context, builder, value=builder.load(flatiter.array))

    flatiter.iternext_specific(context, builder, arrty, arr, result)


@builtin
@implement('getitem', types.Kind(types.NumpyFlatType), types.Kind(types.Integer))
def iternext_numpy_getitem(context, builder, sig, args):
    flatiterty = sig.args[0]
    flatiter, index = args

    flatitercls = make_array_flat_cls(flatiterty)
    flatiter = flatitercls(context, builder, value=flatiter)

    arrty = flatiterty.array_type
    arrcls = context.make_array(arrty)
    arr = arrcls(context, builder, value=builder.load(flatiter.array))

    res = flatiter.getitem(context, builder, arrty, arr, index)
    return impl_ret_borrowed(context, builder, sig.return_type, res)


@builtin
@implement('setitem', types.Kind(types.NumpyFlatType), types.Kind(types.Integer),
           types.Any)
def iternext_numpy_getitem(context, builder, sig, args):
    flatiterty = sig.args[0]
    flatiter, index, value = args

    flatitercls = make_array_flat_cls(flatiterty)
    flatiter = flatitercls(context, builder, value=flatiter)

    arrty = flatiterty.array_type
    arrcls = context.make_array(arrty)
    arr = arrcls(context, builder, value=builder.load(flatiter.array))

    res = flatiter.setitem(context, builder, arrty, arr, index, value)
    return context.get_dummy_value()


@builtin
@implement(numpy.ndenumerate, types.Kind(types.Array))
def make_array_ndenumerate(context, builder, sig, args):
    arrty, = sig.args
    arr, = args
    nditercls = make_array_ndenumerate_cls(types.NumpyNdEnumerateType(arrty))
    nditer = nditercls(context, builder)

    arrayptr = cgutils.alloca_once_value(builder, arr)
    nditer.array = arrayptr

    arrcls = context.make_array(arrty)
    arr = arrcls(context, builder, ref=arrayptr)

    nditer.init_specific(context, builder, arrty, arr)

    res = nditer._getvalue()
    return impl_ret_borrowed(context, builder, sig.return_type, res)


@builtin
@implement('iternext', types.Kind(types.NumpyNdEnumerateType))
@iternext_impl
def iternext_numpy_nditer(context, builder, sig, args, result):
    [nditerty] = sig.args
    [nditer] = args

    nditercls = make_array_ndenumerate_cls(nditerty)
    nditer = nditercls(context, builder, value=nditer)

    arrty = nditerty.array_type
    arrcls = context.make_array(arrty)
    arr = arrcls(context, builder, value=builder.load(nditer.array))

    nditer.iternext_specific(context, builder, arrty, arr, result)


@builtin
@implement(numpy.ndindex, types.VarArg(types.Kind(types.Integer)))
def make_array_ndindex(context, builder, sig, args):
    """ndindex(*shape)"""
    shape = [context.cast(builder, arg, argty, types.intp)
             for argty, arg in zip(sig.args, args)]

    nditercls = make_ndindex_cls(types.NumpyNdIndexType(len(shape)))
    nditer = nditercls(context, builder)
    nditer.init_specific(context, builder, shape)

    res = nditer._getvalue()
    return impl_ret_borrowed(context, builder, sig.return_type, res)

@builtin
@implement(numpy.ndindex, types.Kind(types.BaseTuple))
def make_array_ndindex(context, builder, sig, args):
    """ndindex(shape)"""
    ndim = sig.return_type.ndim
    if ndim > 0:
        idxty = sig.args[0].dtype
        tup = args[0]

        shape = cgutils.unpack_tuple(builder, tup, ndim)
        shape = [context.cast(builder, idx, idxty, types.intp)
                 for idx in shape]
    else:
        shape = []

    nditercls = make_ndindex_cls(types.NumpyNdIndexType(len(shape)))
    nditer = nditercls(context, builder)
    nditer.init_specific(context, builder, shape)

    res = nditer._getvalue()
    return impl_ret_borrowed(context, builder, sig.return_type, res)

@builtin
@implement('iternext', types.Kind(types.NumpyNdIndexType))
@iternext_impl
def iternext_numpy_ndindex(context, builder, sig, args, result):
    [nditerty] = sig.args
    [nditer] = args

    nditercls = make_ndindex_cls(nditerty)
    nditer = nditercls(context, builder, value=nditer)

    nditer.iternext_specific(context, builder, result)


# -----------------------------------------------------------------------------
# Numpy array constructors

def _empty_nd_impl(context, builder, arrtype, shapes):
    """Utility function used for allocating a new array during LLVM code
    generation (lowering).  Given a target context, builder, array
    type, and a tuple or list of lowered dimension sizes, returns a
    LLVM value pointing at a Numba runtime allocated array.
    """
    arycls = make_array(arrtype)
    ary = arycls(context, builder)

    datatype = context.get_data_type(arrtype.dtype)
    itemsize = context.get_constant(types.intp,
                                    context.get_abi_sizeof(datatype))

    # compute array length
    arrlen = context.get_constant(types.intp, 1)
    for s in shapes:
        arrlen = builder.mul(arrlen, s)

    if arrtype.ndim == 0:
        strides = ()
    elif arrtype.layout == 'C':
        strides = [itemsize]
        for dimension_size in reversed(shapes[1:]):
            strides.append(builder.mul(strides[-1], dimension_size))
        strides = tuple(reversed(strides))
    elif arrtype.layout == 'F':
        strides = [itemsize]
        for dimension_size in shapes[:-1]:
            strides.append(builder.mul(strides[-1], dimension_size))
        strides = tuple(strides)
    else:
        raise NotImplementedError(
            "Don't know how to allocate array with layout '{0}'.".format(
                arrtype.layout))

    allocsize = builder.mul(itemsize, arrlen)
    # NOTE: AVX prefer 32-byte alignment
    meminfo = context.nrt_meminfo_alloc_aligned(builder, size=allocsize,
                                                align=32)

    data = context.nrt_meminfo_data(builder, meminfo)

    intp_t = context.get_value_type(types.intp)
    shape_array = cgutils.pack_array(builder, shapes, ty=intp_t)
    strides_array = cgutils.pack_array(builder, strides, ty=intp_t)

    populate_array(ary,
                   data=builder.bitcast(data, datatype.as_pointer()),
                   shape=shape_array,
                   strides=strides_array,
                   itemsize=itemsize,
                   meminfo=meminfo)

    return ary

def _zero_fill_array(context, builder, ary):
    """
    Zero-fill an array.  The array must be contiguous.
    """
    cgutils.memset(builder, ary.data, builder.mul(ary.itemsize, ary.nitems), 0)


def _parse_empty_args(context, builder, sig, args):
    """
    Parse the arguments of a np.empty(), np.zeros() or np.ones() call.
    """
    arrshapetype = sig.args[0]
    arrshape = args[0]
    arrtype = sig.return_type

    if isinstance(arrshapetype, types.Integer):
        ndim = 1
        shapes = [context.cast(builder, arrshape, arrshapetype, types.intp)]
    else:
        ndim = arrshapetype.count
        arrshape = context.cast(builder, arrshape, arrshapetype,
                                types.UniTuple(types.intp, ndim))
        shapes = cgutils.unpack_tuple(builder, arrshape, count=ndim)

    zero = context.get_constant_generic(builder, types.intp, 0)
    for dim in range(ndim):
        is_neg = builder.icmp_signed('<', shapes[dim], zero)
        with cgutils.if_unlikely(builder, is_neg):
            context.call_conv.return_user_exc(builder, ValueError,
                                              ("negative dimensions not allowed",))
    return arrtype, shapes


def _parse_empty_like_args(context, builder, sig, args):
    """
    Parse the arguments of a np.empty_like(), np.zeros_like() or
    np.ones_like() call.
    """
    arytype = sig.args[0]
    if isinstance(arytype, types.Array):
        ary = make_array(arytype)(context, builder, value=args[0])
        shapes = cgutils.unpack_tuple(builder, ary.shape, count=arytype.ndim)
        return sig.return_type, shapes
    else:
        return sig.return_type, ()


@builtin
@implement(numpy.empty, types.Any)
@implement(numpy.empty, types.Any, types.Any)
def numpy_empty_nd(context, builder, sig, args):
    arrtype, shapes = _parse_empty_args(context, builder, sig, args)
    ary = _empty_nd_impl(context, builder, arrtype, shapes)
    return impl_ret_new_ref(context, builder, sig.return_type, ary._getvalue())

@builtin
@implement(numpy.empty_like, types.Any)
@implement(numpy.empty_like, types.Any, types.Kind(types.DTypeSpec))
def numpy_empty_like_nd(context, builder, sig, args):
    arrtype, shapes = _parse_empty_like_args(context, builder, sig, args)
    ary = _empty_nd_impl(context, builder, arrtype, shapes)
    return impl_ret_new_ref(context, builder, sig.return_type, ary._getvalue())


@builtin
@implement(numpy.zeros, types.Any)
@implement(numpy.zeros, types.Any, types.Any)
def numpy_zeros_nd(context, builder, sig, args):
    arrtype, shapes = _parse_empty_args(context, builder, sig, args)
    ary = _empty_nd_impl(context, builder, arrtype, shapes)
    _zero_fill_array(context, builder, ary)
    return impl_ret_new_ref(context, builder, sig.return_type, ary._getvalue())


@builtin
@implement(numpy.zeros_like, types.Any)
@implement(numpy.zeros_like, types.Any, types.Kind(types.DTypeSpec))
def numpy_zeros_like_nd(context, builder, sig, args):
    arrtype, shapes = _parse_empty_like_args(context, builder, sig, args)
    ary = _empty_nd_impl(context, builder, arrtype, shapes)
    _zero_fill_array(context, builder, ary)
    return impl_ret_new_ref(context, builder, sig.return_type, ary._getvalue())


if numpy_version >= (1, 8):
    @builtin
    @implement(numpy.full, types.Any, types.Any)
    def numpy_full_nd(context, builder, sig, args):

        def full(shape, value):
            arr = numpy.empty(shape)
            for idx in numpy.ndindex(arr.shape):
                arr[idx] = value
            return arr

        res = context.compile_internal(builder, full, sig, args)
        return impl_ret_new_ref(context, builder, sig.return_type, res)

    @builtin
    @implement(numpy.full, types.Any, types.Any, types.Kind(types.DTypeSpec))
    def numpy_full_dtype_nd(context, builder, sig, args):

        def full(shape, value, dtype):
            arr = numpy.empty(shape, dtype)
            for idx in numpy.ndindex(arr.shape):
                arr[idx] = value
            return arr

        res = context.compile_internal(builder, full, sig, args)
        return impl_ret_new_ref(context, builder, sig.return_type, res)


    @builtin
    @implement(numpy.full_like, types.Any, types.Any)
    def numpy_full_like_nd(context, builder, sig, args):

        def full_like(arr, value):
            arr = numpy.empty_like(arr)
            for idx in numpy.ndindex(arr.shape):
                arr[idx] = value
            return arr

        res = context.compile_internal(builder, full_like, sig, args)
        return impl_ret_new_ref(context, builder, sig.return_type, res)


    @builtin
    @implement(numpy.full_like, types.Any, types.Any, types.Kind(types.DTypeSpec))
    def numpy_full_like_nd(context, builder, sig, args):

        def full_like(arr, value, dtype):
            arr = numpy.empty_like(arr, dtype)
            for idx in numpy.ndindex(arr.shape):
                arr[idx] = value
            return arr

        res = context.compile_internal(builder, full_like, sig, args)
        return impl_ret_new_ref(context, builder, sig.return_type, res)


@builtin
@implement(numpy.ones, types.Any)
def numpy_ones_nd(context, builder, sig, args):

    def ones(shape):
        arr = numpy.empty(shape)
        for idx in numpy.ndindex(arr.shape):
            arr[idx] = 1
        return arr

    valty = sig.return_type.dtype
    res = context.compile_internal(builder, ones, sig, args,
                                   locals={'c': valty})
    return impl_ret_new_ref(context, builder, sig.return_type, res)

@builtin
@implement(numpy.ones, types.Any, types.Kind(types.DTypeSpec))
def numpy_ones_dtype_nd(context, builder, sig, args):

    def ones(shape, dtype):
        arr = numpy.empty(shape, dtype)
        for idx in numpy.ndindex(arr.shape):
            arr[idx] = 1
        return arr

    res = context.compile_internal(builder, ones, sig, args)
    return impl_ret_new_ref(context, builder, sig.return_type, res)

@builtin
@implement(numpy.ones_like, types.Any)
def numpy_ones_like_nd(context, builder, sig, args):

    def ones_like(arr):
        arr = numpy.empty_like(arr)
        for idx in numpy.ndindex(arr.shape):
            arr[idx] = 1
        return arr

    res = context.compile_internal(builder, ones_like, sig, args)
    return impl_ret_new_ref(context, builder, sig.return_type, res)

@builtin
@implement(numpy.ones_like, types.Any, types.Kind(types.DTypeSpec))
def numpy_ones_like_dtype_nd(context, builder, sig, args):

    def ones_like(arr, dtype):
        arr = numpy.empty_like(arr, dtype)
        for idx in numpy.ndindex(arr.shape):
            arr[idx] = 1
        return arr

    res = context.compile_internal(builder, ones_like, sig, args)
    return impl_ret_new_ref(context, builder, sig.return_type, res)


@builtin
@implement(numpy.identity, types.Kind(types.Integer))
def numpy_identity(context, builder, sig, args):

    def identity(n):
        arr = numpy.zeros((n, n))
        for i in range(n):
            arr[i, i] = 1
        return arr

    res = context.compile_internal(builder, identity, sig, args)
    return impl_ret_new_ref(context, builder, sig.return_type, res)

@builtin
@implement(numpy.identity, types.Kind(types.Integer), types.Kind(types.DTypeSpec))
def numpy_identity(context, builder, sig, args):

    def identity(n, dtype):
        arr = numpy.zeros((n, n), dtype)
        for i in range(n):
            arr[i, i] = 1
        return arr

    res = context.compile_internal(builder, identity, sig, args)
    return impl_ret_new_ref(context, builder, sig.return_type, res)


@builtin
@implement(numpy.eye, types.Kind(types.Integer))
def numpy_eye(context, builder, sig, args):

    def eye(n):
        return numpy.identity(n)

    res = context.compile_internal(builder, eye, sig, args)
    return impl_ret_new_ref(context, builder, sig.return_type, res)

@builtin
@implement(numpy.eye, types.Kind(types.Integer), types.Kind(types.Integer))
def numpy_eye(context, builder, sig, args):

    def eye(n, m):
        return numpy.eye(n, m, 0, numpy.float64)

    res = context.compile_internal(builder, eye, sig, args)
    return impl_ret_new_ref(context, builder, sig.return_type, res)

@builtin
@implement(numpy.eye, types.Kind(types.Integer), types.Kind(types.Integer),
           types.Kind(types.Integer))
def numpy_eye(context, builder, sig, args):

    def eye(n, m, k):
        return numpy.eye(n, m, k, numpy.float64)

    res = context.compile_internal(builder, eye, sig, args)
    return impl_ret_new_ref(context, builder, sig.return_type, res)

@builtin
@implement(numpy.eye, types.Kind(types.Integer), types.Kind(types.Integer),
           types.Kind(types.Integer), types.Kind(types.DTypeSpec))
def numpy_eye(context, builder, sig, args):

    def eye(n, m, k, dtype):
        arr = numpy.zeros((n, m), dtype)
        if k >= 0:
            d = min(n, m - k)
            for i in range(d):
                arr[i, i + k] = 1
        else:
            d = min(n + k, m)
            for i in range(d):
                arr[i - k, i] = 1
        return arr

    res = context.compile_internal(builder, eye, sig, args)
    return impl_ret_new_ref(context, builder, sig.return_type, res)


@builtin
@implement(numpy.arange, types.Kind(types.Number))
def numpy_arange_1(context, builder, sig, args):
    dtype = as_dtype(sig.return_type.dtype)

    def arange(stop):
        return numpy.arange(0, stop, 1, dtype)

    res = context.compile_internal(builder, arange, sig, args)
    return impl_ret_new_ref(context, builder, sig.return_type, res)

@builtin
@implement(numpy.arange, types.Kind(types.Number), types.Kind(types.Number))
def numpy_arange_2(context, builder, sig, args):
    dtype = as_dtype(sig.return_type.dtype)

    def arange(start, stop):
        return numpy.arange(start, stop, 1, dtype)

    res = context.compile_internal(builder, arange, sig, args)
    return impl_ret_new_ref(context, builder, sig.return_type, res)


@builtin
@implement(numpy.arange, types.Kind(types.Number), types.Kind(types.Number),
           types.Kind(types.Number))
def numpy_arange_3(context, builder, sig, args):
    dtype = as_dtype(sig.return_type.dtype)

    def arange(start, stop, step):
        return numpy.arange(start, stop, step, dtype)

    res = context.compile_internal(builder, arange, sig, args)
    return impl_ret_new_ref(context, builder, sig.return_type, res)

@builtin
@implement(numpy.arange, types.Kind(types.Number), types.Kind(types.Number),
           types.Kind(types.Number), types.Kind(types.DTypeSpec))
def numpy_arange_4(context, builder, sig, args):

    if any(isinstance(a, types.Complex) for a in sig.args):
        def arange(start, stop, step, dtype):
            nitems_c = (stop - start) / step
            nitems_r = math.ceil(nitems_c.real)
            nitems_i = math.ceil(nitems_c.imag)
            nitems = max(min(nitems_i, nitems_r), 0)
            arr = numpy.empty(nitems, dtype)
            val = start
            for i in range(nitems):
                arr[i] = val
                val += step
            return arr
    else:
        def arange(start, stop, step, dtype):
            nitems_r = math.ceil((stop - start) / step)
            nitems = max(nitems_r, 0)
            arr = numpy.empty(nitems, dtype)
            val = start
            for i in range(nitems):
                arr[i] = val
                val += step
            return arr

    res = context.compile_internal(builder, arange, sig, args,
                                   locals={'nitems': types.intp})
    return impl_ret_new_ref(context, builder, sig.return_type, res)

@builtin
@implement(numpy.linspace, types.Kind(types.Number), types.Kind(types.Number))
def numpy_linspace_2(context, builder, sig, args):

    def linspace(start, stop):
        return numpy.linspace(start, stop, 50)

    res = context.compile_internal(builder, linspace, sig, args)
    return impl_ret_new_ref(context, builder, sig.return_type, res)

@builtin
@implement(numpy.linspace, types.Kind(types.Number), types.Kind(types.Number),
           types.Kind(types.Integer))
def numpy_linspace_3(context, builder, sig, args):
    dtype = as_dtype(sig.return_type.dtype)

    def linspace(start, stop, num):
        arr = numpy.empty(num, dtype)
        div = num - 1
        delta = stop - start
        arr[0] = start
        for i in range(1, num):
            arr[i] = start + delta * (i / div)
        return arr

    res = context.compile_internal(builder, linspace, sig, args)
    return impl_ret_new_ref(context, builder, sig.return_type, res)


@builtin
@implement("array.copy", types.Kind(types.Array))
def array_copy(context, builder, sig, args):
    arytype = sig.args[0]
    ary = make_array(arytype)(context, builder, value=args[0])
    shapes = cgutils.unpack_tuple(builder, ary.shape)

    rettype = sig.return_type
    ret = _empty_nd_impl(context, builder, rettype, shapes)

    src_data = ary.data
    dest_data = ret.data

    assert rettype.layout == "C"
    if arytype.layout == "C":
        # Fast path: memcpy
        # Compute array length
        arrlen = context.get_constant(types.intp, 1)
        for s in shapes:
            arrlen = builder.mul(arrlen, s)
        arrlen = builder.mul(arrlen, ary.itemsize)

        pchar = lc.Type.int(8).as_pointer()
        memcpy = builder.module.declare_intrinsic(
            'llvm.memcpy', [pchar, pchar, arrlen.type])
        builder.call(memcpy,
                     (builder.bitcast(dest_data, pchar),
                      builder.bitcast(src_data, pchar),
                      arrlen,
                      lc.Constant.int(lc.Type.int(32), 0),
                      lc.Constant.int(lc.Type.int(1), 0),
                      ))

    else:
        src_strides = cgutils.unpack_tuple(builder, ary.strides)
        dest_strides = cgutils.unpack_tuple(builder, ret.strides)
        intp_t = context.get_value_type(types.intp)

        with cgutils.loop_nest(builder, shapes, intp_t) as indices:
            src_ptr = cgutils.get_item_pointer2(builder, src_data,
                                                shapes, src_strides,
                                                arytype.layout, indices)
            dest_ptr = cgutils.get_item_pointer2(builder, dest_data,
                                                 shapes, dest_strides,
                                                 rettype.layout, indices)
            builder.store(builder.load(src_ptr), dest_ptr)

    return impl_ret_new_ref(context, builder, sig.return_type, ret._getvalue())


@builtin
@implement(numpy.frombuffer, types.Kind(types.Buffer))
@implement(numpy.frombuffer, types.Kind(types.Buffer), types.Kind(types.DTypeSpec))
def np_frombuffer(context, builder, sig, args):
    bufty = sig.args[0]
    aryty = sig.return_type

    buf = make_array(bufty)(context, builder, value=args[0])
    out_ary_ty = make_array(aryty)
    out_ary = out_ary_ty(context, builder)
    out_datamodel = out_ary._datamodel

    itemsize = get_itemsize(context, aryty)
    ll_itemsize = lc.Constant.int(buf.itemsize.type, itemsize)
    nbytes = builder.mul(buf.nitems, buf.itemsize)

    # Check that the buffer size is compatible
    rem = builder.srem(nbytes, ll_itemsize)
    is_incompatible = cgutils.is_not_null(builder, rem)
    with builder.if_then(is_incompatible, likely=False):
        msg = "buffer size must be a multiple of element size"
        context.call_conv.return_user_exc(builder, ValueError, (msg,))

    shape = cgutils.pack_array(builder, [builder.sdiv(nbytes, ll_itemsize)])
    strides = cgutils.pack_array(builder, [ll_itemsize])
    data = builder.bitcast(buf.data,
                           context.get_value_type(out_datamodel.get_type('data')))

    populate_array(out_ary,
                   data=data,
                   shape=shape,
                   strides=strides,
                   itemsize=ll_itemsize,
                   meminfo=buf.meminfo,
                   parent=buf.parent,)

    res = out_ary._getvalue()
    return impl_ret_borrowed(context, builder, sig.return_type, res)


# -----------------------------------------------------------------------------
# Sorting

_sorting_init = False

def lt_floats(a, b):
    return math.isnan(b) or a < b

def load_sorts():
    """
    Load quicksort lazily, to avoid circular imports accross the jit() global.
    """
    g = globals()
    if g['_sorting_init']:
        return

    default_quicksort = quicksort.make_jit_quicksort()
    g['run_default_quicksort'] = default_quicksort.run_quicksort
    float_quicksort = quicksort.make_jit_quicksort(lt=lt_floats)
    g['run_float_quicksort'] = float_quicksort.run_quicksort
    g['_sorting_init'] = True


@builtin
@implement("array.sort", types.Kind(types.Array))
def array_sort(context, builder, sig, args):
    load_sorts()

    arytype = sig.args[0]
    dtype = arytype.dtype

    if isinstance(dtype, types.Float):
        def array_sort_impl(arr):
            return run_float_quicksort(arr)
    else:
        def array_sort_impl(arr):
            return run_default_quicksort(arr)

    return context.compile_internal(builder, array_sort_impl, sig, args)


@builtin
@implement(numpy.sort, types.Kind(types.Array))
def np_sort(context, builder, sig, args):

    def np_sort_impl(a):
        res = a.copy()
        res.sort()
        return res

    return context.compile_internal(builder, np_sort_impl, sig, args)
