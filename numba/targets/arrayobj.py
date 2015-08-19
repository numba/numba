"""
Implementation of operations on Array objects and objects supporting
the buffer protocol.
"""

from __future__ import print_function, absolute_import, division

import math

import llvmlite.llvmpy.core as lc

import numpy
from llvmlite.llvmpy.core import Constant
from numba import types, cgutils, typing
from numba.numpy_support import as_dtype
from numba.numpy_support import version as numpy_version
from numba.targets.imputils import (builtin, builtin_attr, implement,
                                    impl_attribute, impl_attribute_generic,
                                    iternext_impl, struct_factory,
                                    impl_ret_borrowed, impl_ret_new_ref,
                                    impl_ret_untracked)
from numba.typing import signature
from . import slicing


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


def populate_array(array, data, shape, strides, itemsize, meminfo,
                   parent=None):
    """
    Helper function for populating array structures.
    This avoids forgetting to set fields.
    """
    context = array._context
    builder = array._builder
    datamodel = array._datamodel
    required_fields = set(datamodel._fields)

    if meminfo is None:
        meminfo = Constant.null(context.get_value_type(
            datamodel.get_type('meminfo')))

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


@struct_factory(types.ArrayIterator)
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
    ptr = cgutils.get_item_pointer(builder, arrayty, array, [idx],
                                   wraparound=wraparound)
    return context.unpack_value(builder, arrayty.dtype, ptr)

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

@builtin
@implement('getitem', types.Kind(types.Buffer), types.Kind(types.Integer))
def getitem_arraynd_intp(context, builder, sig, args):
    aryty, idxty = sig.args
    ary, idx = args
    arystty = make_array(aryty)
    adapted_ary = arystty(context, builder, ary)
    ndim = aryty.ndim
    if ndim == 1:
        # Return a value
        result = _getitem_array1d(context, builder, aryty, adapted_ary, idx,
                                  wraparound=idxty.signed)
    elif ndim > 1:
        # Return a subview over the array
        out_ary_ty = make_array(aryty.copy(ndim = ndim - 1))
        out_ary = out_ary_ty(context, builder)
        in_shapes = cgutils.unpack_tuple(builder, adapted_ary.shape, count=ndim)
        in_strides = cgutils.unpack_tuple(builder, adapted_ary.strides,
                                          count=ndim)
        data_p = cgutils.get_item_pointer2(builder, adapted_ary.data, in_shapes,
                                           in_strides, aryty.layout, [idx],
                                           wraparound=idxty.signed)
        populate_array(out_ary,
                       data=data_p,
                       shape=cgutils.pack_array(builder, in_shapes[1:]),
                       strides=cgutils.pack_array(builder, in_strides[1:]),
                       itemsize=adapted_ary.itemsize,
                       meminfo=adapted_ary.meminfo,
                       parent=adapted_ary.parent,)

        result = out_ary._getvalue()
    else:
        raise NotImplementedError("1D indexing into %dD array" % aryty.ndim)
    return impl_ret_borrowed(context, builder, sig.return_type, result)


def _getitem_array_generic(context, builder, return_type, aryty, ary,
                           index_types, indices):
    """
    Return the result of indexing *ary* with the given *indices*.
    """
    shapes = cgutils.unpack_tuple(builder, ary.shape, aryty.ndim)
    strides = cgutils.unpack_tuple(builder, ary.strides, aryty.ndim)

    output_indices = []
    output_shapes = []
    output_strides = []

    for ax, (indexval, idxty) in enumerate(zip(indices, index_types)):
        if idxty == types.slice3_type:
            slice = slicing.Slice(context, builder, value=indexval)
            slicing.fix_slice(builder, slice, shapes[ax])
            output_indices.append(slice.start)
            sh = slicing.get_slice_length(builder, slice)
            st = slicing.fix_stride(builder, slice, strides[ax])
            output_shapes.append(sh)
            output_strides.append(st)
        else:
            assert isinstance(idxty, types.Integer)
            ind = context.cast(builder, indexval, idxty, types.intp)
            ind = slicing.fix_index(builder, ind, shapes[ax])
            output_indices.append(ind)

    dataptr = cgutils.get_item_pointer(builder, aryty, ary,
                                       output_indices,
                                       wraparound=False)
    # Build array
    retary = make_array(return_type)(context, builder)
    populate_array(retary,
                   data=dataptr,
                   shape=cgutils.pack_array(builder, output_shapes),
                   strides=cgutils.pack_array(builder, output_strides),
                   itemsize=ary.itemsize,
                   meminfo=ary.meminfo,
                   parent=ary.parent)
    return retary._getvalue()


@builtin
@implement('getitem', types.Kind(types.Buffer), types.slice3_type)
def getitem_array1d_slice(context, builder, sig, args):
    aryty, idxty = sig.args
    ary, idx = args

    if aryty.ndim != 1:
        # TODO
        raise NotImplementedError("1D indexing into %dD array" % aryty.ndim)

    ary = make_array(aryty)(context, builder, value=ary)

    res = _getitem_array_generic(context, builder, sig.return_type,
                                 aryty, ary, (idxty,), (idx,))
    return impl_ret_borrowed(context, builder, sig.return_type, res)


@builtin
@implement('getitem', types.Kind(types.Buffer), types.Kind(types.BaseTuple))
def getitem_array_tuple(context, builder, sig, args):
    aryty, tupty = sig.args
    ary, tup = args

    arystty = make_array(aryty)
    ary = arystty(context, builder, ary)

    ndim = aryty.ndim
    if isinstance(sig.return_type, types.Array):
        # Generic array slicing
        tup_items = cgutils.unpack_tuple(builder, tup, ndim)
        res = _getitem_array_generic(context, builder, sig.return_type,
                                     aryty, ary, tupty, tup_items)
    else:
        # Plain indexing (returning a scalar)
        indices = cgutils.unpack_tuple(builder, tup, count=len(tupty))
        indices = [context.cast(builder, i, t, types.intp)
                   for t, i in zip(tupty, indices)]
        ptr = cgutils.get_item_pointer(builder, aryty, ary, indices,
                                       wraparound=True)

        res = context.unpack_value(builder, aryty.dtype, ptr)

    return impl_ret_borrowed(context, builder ,sig.return_type, res)


@builtin
@implement('setitem', types.Kind(types.Buffer), types.Kind(types.Integer),
           types.Any)
def setitem_array1d(context, builder, sig, args):
    aryty, idxty, valty = sig.args
    ary, idx, val = args

    arystty = make_array(aryty)
    ary = arystty(context, builder, ary)

    ptr = cgutils.get_item_pointer(builder, aryty, ary, [idx],
                                   wraparound=idxty.signed)

    val = context.cast(builder, val, valty, aryty.dtype)

    context.pack_value(builder, aryty.dtype, val, ptr)


@builtin
@implement('setitem', types.Kind(types.Buffer),
           types.Kind(types.UniTuple), types.Any)
def setitem_array_unituple(context, builder, sig, args):
    aryty, idxty, valty = sig.args
    ary, idx, val = args

    arystty = make_array(aryty)
    ary = arystty(context, builder, ary)

    # TODO: other than layout
    indices = cgutils.unpack_tuple(builder, idx, count=len(idxty))
    indices = [context.cast(builder, i, t, types.intp)
               for t, i in zip(idxty, indices)]
    ptr = cgutils.get_item_pointer(builder, aryty, ary, indices,
                                   wraparound=idxty.dtype.signed)
    context.pack_value(builder, aryty.dtype, val, ptr)


@builtin
@implement('setitem', types.Kind(types.Buffer),
           types.Kind(types.Tuple), types.Any)
def setitem_array_tuple(context, builder, sig, args):
    aryty, idxty, valty = sig.args
    ary, idx, val = args

    arystty = make_array(aryty)
    ary = arystty(context, builder, ary)

    # TODO: other than layout
    indices = cgutils.unpack_tuple(builder, idx, count=len(idxty))
    indices = [context.cast(builder, i, t, types.intp)
               for t, i in zip(idxty, indices)]
    ptr = cgutils.get_item_pointer(builder, aryty, ary, indices,
                                   wraparound=True)
    context.pack_value(builder, aryty.dtype, val, ptr)

@builtin
@implement('setitem', types.Kind(types.Buffer),
           types.slice3_type, types.Any)
def setitem_array1d_slice(context, builder, sig, args):
    aryty, idxty, valty = sig.args
    ary, idx, val = args
    arystty = make_array(aryty)
    ary = arystty(context, builder, ary)
    shapes = cgutils.unpack_tuple(builder, ary.shape, aryty.ndim)
    slicestruct = slicing.Slice(context, builder, value=idx)

    # the logic here follows that of Python's Objects/sliceobject.c
    # in particular PySlice_GetIndicesEx function
    ZERO = Constant.int(slicestruct.step.type, 0)
    NEG_ONE = Constant.int(slicestruct.start.type, -1)

    b_step_eq_zero = builder.icmp(lc.ICMP_EQ, slicestruct.step, ZERO)
    # bail if step is 0
    with builder.if_then(b_step_eq_zero):
        context.call_conv.return_user_exc(builder, ValueError,
                                          ("slice step cannot be zero",))

    # adjust for negative indices for start
    start = cgutils.alloca_once_value(builder, slicestruct.start)
    b_start_lt_zero = builder.icmp(lc.ICMP_SLT, builder.load(start), ZERO)
    with builder.if_then(b_start_lt_zero):
        add = builder.add(builder.load(start), shapes[0])
        builder.store(add, start)

    b_start_lt_zero = builder.icmp(lc.ICMP_SLT, builder.load(start), ZERO)
    with builder.if_then(b_start_lt_zero):
        b_step_lt_zero = builder.icmp(lc.ICMP_SLT, slicestruct.step, ZERO)
        cond = builder.select(b_step_lt_zero, NEG_ONE, ZERO)
        builder.store(cond, start)

    b_start_geq_len = builder.icmp(lc.ICMP_SGE, builder.load(start), shapes[0])
    ONE = Constant.int(shapes[0].type, 1)
    with builder.if_then(b_start_geq_len):
        b_step_lt_zero = builder.icmp(lc.ICMP_SLT, slicestruct.step, ZERO)
        cond = builder.select(b_step_lt_zero, builder.sub(shapes[0], ONE), shapes[0])
        builder.store(cond, start)

    # adjust stop for negative value
    stop = cgutils.alloca_once_value(builder, slicestruct.stop)
    b_stop_lt_zero = builder.icmp(lc.ICMP_SLT, builder.load(stop), ZERO)
    with builder.if_then(b_stop_lt_zero):
        add = builder.add(builder.load(stop), shapes[0])
        builder.store(add, stop)

    b_stop_lt_zero = builder.icmp(lc.ICMP_SLT, builder.load(stop), ZERO)
    with builder.if_then(b_stop_lt_zero):
        b_step_lt_zero = builder.icmp(lc.ICMP_SLT, slicestruct.step, ZERO)
        cond = builder.select(b_step_lt_zero, NEG_ONE, ZERO)
        builder.store(cond, start)

    b_stop_geq_len = builder.icmp(lc.ICMP_SGE, builder.load(stop), shapes[0])
    ONE = Constant.int(shapes[0].type, 1)
    with builder.if_then(b_stop_geq_len):
        b_step_lt_zero = builder.icmp(lc.ICMP_SLT, slicestruct.step, ZERO)
        cond = builder.select(b_step_lt_zero, builder.sub(shapes[0], ONE), shapes[0])
        builder.store(cond, stop)

    with cgutils.for_range_slice_generic(builder, builder.load(start),
                                         builder.load(stop),
                                         slicestruct.step) as (pos_range, neg_range):
        with pos_range as (loop_idx1, _):
            ptr = cgutils.get_item_pointer(builder, aryty, ary,
                                [loop_idx1],
                                wraparound=True)
            context.pack_value(builder, aryty.dtype, val, ptr)
        with neg_range as (loop_idx2, _):
            ptr = cgutils.get_item_pointer(builder, aryty, ary,
                                [loop_idx2],
                                wraparound=True)
            context.pack_value(builder, aryty.dtype, val, ptr)


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
# Computations

@builtin
@implement(numpy.sum, types.Kind(types.Array))
@implement("array.sum", types.Kind(types.Array))
def array_sum(context, builder, sig, args):
    zero = sig.return_type(0)

    def array_sum_impl(arr):
        c = zero
        for v in arr.flat:
            c += v
        return c

    res = context.compile_internal(builder, array_sum_impl, sig, args,
                                    locals=dict(c=sig.return_type))
    return impl_ret_borrowed(context, builder, sig.return_type, res)

@builtin
@implement(numpy.prod, types.Kind(types.Array))
@implement("array.prod", types.Kind(types.Array))
def array_prod(context, builder, sig, args):

    def array_prod_impl(arr):
        c = 1
        for v in arr.flat:
            c *= v
        return c

    res = context.compile_internal(builder, array_prod_impl, sig, args,
                                    locals=dict(c=sig.return_type))
    return impl_ret_borrowed(context, builder, sig.return_type, res)

@builtin
@implement(numpy.cumsum, types.Kind(types.Array))
@implement("array.cumsum", types.Kind(types.Array))
def array_cumsum(context, builder, sig, args):
    scalar_dtype = sig.return_type.dtype
    dtype = as_dtype(scalar_dtype)
    zero = scalar_dtype(0)

    def array_cumsum_impl(arr):
        size = 1
        for i in arr.shape:
            size = size * i
        out = numpy.empty(size, dtype)
        c = zero
        for idx, v in enumerate(arr.flat):
            c += v
            out[idx] = c
        return out

    res = context.compile_internal(builder, array_cumsum_impl, sig, args,
                                   locals=dict(c=scalar_dtype))
    return impl_ret_new_ref(context, builder, sig.return_type, res)



@builtin
@implement(numpy.cumprod, types.Kind(types.Array))
@implement("array.cumprod", types.Kind(types.Array))
def array_cumprod(context, builder, sig, args):
    scalar_dtype = sig.return_type.dtype
    dtype = as_dtype(scalar_dtype)

    def array_cumprod_impl(arr):
        size = 1
        for i in arr.shape:
            size = size * i
        out = numpy.empty(size, dtype)
        c = 1
        for idx, v in enumerate(arr.flat):
            c *= v
            out[idx] = c
        return out

    res = context.compile_internal(builder, array_cumprod_impl, sig, args,
                                   locals=dict(c=scalar_dtype))
    return impl_ret_new_ref(context, builder, sig.return_type, res)

@builtin
@implement(numpy.mean, types.Kind(types.Array))
@implement("array.mean", types.Kind(types.Array))
def array_mean(context, builder, sig, args):
    zero = sig.return_type(0)

    def array_mean_impl(arr):
        # Can't use the naive `arr.sum() / arr.size`, as it would return
        # a wrong result on integer sum overflow.
        c = zero
        for v in arr.flat:
            c += v
        return c / arr.size

    res = context.compile_internal(builder, array_mean_impl, sig, args,
                                   locals=dict(c=sig.return_type))
    return impl_ret_untracked(context, builder, sig.return_type, res)

@builtin
@implement(numpy.var, types.Kind(types.Array))
@implement("array.var", types.Kind(types.Array))
def array_var(context, builder, sig, args):
    def array_var_impl(arry):
        # Compute the mean
        m = arry.mean()

        # Compute the sum of square diffs
        ssd = 0
        for v in arry.flat:
            ssd += (v - m) ** 2
        return ssd / arry.size

    res = context.compile_internal(builder, array_var_impl, sig, args)
    return impl_ret_untracked(context, builder, sig.return_type, res)


@builtin
@implement(numpy.std, types.Kind(types.Array))
@implement("array.std", types.Kind(types.Array))
def array_std(context, builder, sig, args):
    def array_std_impl(arry):
        return arry.var() ** 0.5
    res = context.compile_internal(builder, array_std_impl, sig, args)
    return impl_ret_untracked(context, builder, sig.return_type, res)


@builtin
@implement(numpy.min, types.Kind(types.Array))
@implement("array.min", types.Kind(types.Array))
def array_min(context, builder, sig, args):
    ty = sig.args[0].dtype
    if isinstance(ty, (types.NPDatetime, types.NPTimedelta)):
        # NaT is smaller than every other value, but it is
        # ignored as far as min() is concerned.
        nat = ty('NaT')

        def array_min_impl(arry):
            min_value = nat
            it = arry.flat
            for v in it:
                if v != nat:
                    min_value = v
                    break

            for v in it:
                if v != nat and v < min_value:
                    min_value = v
            return min_value

    else:
        def array_min_impl(arry):
            for v in arry.flat:
                min_value = v
                break

            for v in arry.flat:
                if v < min_value:
                    min_value = v
            return min_value
    res = context.compile_internal(builder, array_min_impl, sig, args)
    return impl_ret_borrowed(context, builder, sig.return_type, res)


@builtin
@implement(numpy.max, types.Kind(types.Array))
@implement("array.max", types.Kind(types.Array))
def array_max(context, builder, sig, args):
    def array_max_impl(arry):
        for v in arry.flat:
            max_value = v
            break

        for v in arry.flat:
            if v > max_value:
                max_value = v
        return max_value
    res = context.compile_internal(builder, array_max_impl, sig, args)
    return impl_ret_borrowed(context, builder, sig.return_type, res)


@builtin
@implement(numpy.argmin, types.Kind(types.Array))
@implement("array.argmin", types.Kind(types.Array))
def array_argmin(context, builder, sig, args):
    # XXX argmin() is inconsistent with min() on NaT values:
    # https://github.com/numpy/numpy/issues/6030
    def array_argmin_impl(arry):
        for v in arry.flat:
            min_value = v
            min_idx = 0
            break

        idx = 0
        for v in arry.flat:
            if v < min_value:
                min_value = v
                min_idx = idx
            idx += 1
        return min_idx
    res = context.compile_internal(builder, array_argmin_impl, sig, args)
    return impl_ret_untracked(context, builder, sig.return_type, res)


@builtin
@implement(numpy.argmax, types.Kind(types.Array))
@implement("array.argmax", types.Kind(types.Array))
def array_argmax(context, builder, sig, args):
    def array_argmax_impl(arry):
        for v in arry.flat:
            max_value = v
            max_idx = 0
            break

        idx = 0
        for v in arry.flat:
            if v > max_value:
                max_value = v
                max_idx = idx
            idx += 1
        return max_idx
    res = context.compile_internal(builder, array_argmax_impl, sig, args)
    return impl_ret_untracked(context, builder, sig.return_type, res)


@builtin
@implement(numpy.median, types.Kind(types.Array))
def array_median(context, builder, sig, args):

    def partition(A, low, high):
        mid = (low+high) // 2
        # median of three {low, middle, high}
        LM = A[low] <= A[mid]
        MH = A[mid] <= A[high]
        LH = A[low] <= A[high]

        if LM == MH:
            median3 = mid
        elif LH != LM:
            median3 = low
        else:
            median3 = high

        # choose median3 as the pivot
        A[high], A[median3] = A[median3], A[high]

        x = A[high]
        i = low
        for j in range(low, high):
            if A[j] <= x:
                A[i], A[j] = A[j], A[i]
                i += 1
        A[i], A[high] = A[high], A[i]
        return i

    sig_partition = typing.signature(types.intp, *(sig.args[0], types.intp, types.intp))
    _partition = context.compile_subroutine(builder, partition, sig_partition)

    def select(arry, k):
        n = arry.shape[0]
        # XXX: assuming flat array till array.flatten is implemented
        # temp_arry = arry.flatten()
        temp_arry = arry.copy()
        high = n-1
        low = 0
        # NOTE: high is inclusive
        i = _partition(temp_arry, low, high)
        while i != k:
            if i < k:
                low = i+1
                i = _partition(temp_arry, low, high)
            else:
                high = i-1
                i = _partition(temp_arry, low, high)
        return temp_arry[k]

    sig_select = typing.signature(sig.args[0].dtype, *(sig.args[0], types.intp))
    _select = context.compile_subroutine(builder, select, sig_select)

    def median(arry):
        n = arry.shape[0]
        if n % 2 == 0:
            return (_select(arry, n//2 - 1) + _select(arry, n//2))/2
        else:
            return _select(arry, n//2)

    res = context.compile_internal(builder, median, sig, args)
    return impl_ret_untracked(context, builder, sig.return_type, res)


def _np_round_intrinsic(tp):
    # np.round() always rounds half to even
    return "llvm.rint.f%d" % (tp.bitwidth,)

def _np_round_float(context, builder, tp, val):
    llty = context.get_value_type(tp)
    module = builder.module
    fnty = lc.Type.function(llty, [llty])
    fn = module.get_or_insert_function(fnty, name=_np_round_intrinsic(tp))
    return builder.call(fn, (val,))

@builtin
@implement(numpy.round, types.Kind(types.Float))
def scalar_round_unary(context, builder, sig, args):
    res =  _np_round_float(context, builder, sig.args[0], args[0])
    return impl_ret_untracked(context, builder, sig.return_type, res)

@builtin
@implement(numpy.round, types.Kind(types.Integer))
def scalar_round_unary(context, builder, sig, args):
    res = args[0]
    return impl_ret_untracked(context, builder, sig.return_type, res)

@builtin
@implement(numpy.round, types.Kind(types.Complex))
def scalar_round_unary_complex(context, builder, sig, args):
    fltty = sig.args[0].underlying_float
    cplx_cls = context.make_complex(sig.args[0])
    z = cplx_cls(context, builder, args[0])
    z.real = _np_round_float(context, builder, fltty, z.real)
    z.imag = _np_round_float(context, builder, fltty, z.imag)
    res = z._getvalue()
    return impl_ret_untracked(context, builder, sig.return_type, res)

@builtin
@implement(numpy.round, types.Kind(types.Float), types.Kind(types.Integer))
@implement(numpy.round, types.Kind(types.Integer), types.Kind(types.Integer))
def scalar_round_binary_float(context, builder, sig, args):
    def round_ndigits(x, ndigits):
        if math.isinf(x) or math.isnan(x):
            return x

        # NOTE: this is CPython's algorithm, but perhaps this is overkill
        # when emulating Numpy's behaviour.
        if ndigits >= 0:
            if ndigits > 22:
                # pow1 and pow2 are each safe from overflow, but
                # pow1*pow2 ~= pow(10.0, ndigits) might overflow.
                pow1 = 10.0 ** (ndigits - 22)
                pow2 = 1e22
            else:
                pow1 = 10.0 ** ndigits
                pow2 = 1.0
            y = (x * pow1) * pow2
            if math.isinf(y):
                return x
            return (numpy.round(y) / pow2) / pow1

        else:
            pow1 = 10.0 ** (-ndigits)
            y = x / pow1
            return numpy.round(y) * pow1

    res = context.compile_internal(builder, round_ndigits, sig, args)
    return impl_ret_untracked(context, builder, sig.return_type, res)

@builtin
@implement(numpy.round, types.Kind(types.Complex), types.Kind(types.Integer))
def scalar_round_binary_complex(context, builder, sig, args):
    def round_ndigits(z, ndigits):
        return complex(numpy.round(z.real, ndigits),
                       numpy.round(z.imag, ndigits))

    res = context.compile_internal(builder, round_ndigits, sig, args)
    return impl_ret_untracked(context, builder, sig.return_type, res)


@builtin
@implement(numpy.round, types.Kind(types.Array), types.Kind(types.Integer),
           types.Kind(types.Array))
def array_round(context, builder, sig, args):
    def array_round_impl(arr, decimals, out):
        if arr.shape != out.shape:
            raise ValueError("invalid output shape")
        for index, val in numpy.ndenumerate(arr):
            out[index] = numpy.round(val, decimals)
        return out

    res = context.compile_internal(builder, array_round_impl, sig, args)
    return impl_ret_new_ref(context, builder, sig.return_type, res)


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


@builtin_attr
@impl_attribute_generic(types.Kind(types.Array))
def array_record_getattr(context, builder, typ, value, attr):
    """
    Generic getattr() implementation for record arrays: fetch the given
    record member.
    """
    arrayty = make_array(typ)
    array = arrayty(context, builder, value)

    rectype = typ.dtype
    if not isinstance(rectype, types.Record):
        raise AttributeError("attribute %r of %s not defined" % (attr, typ))
    dtype = rectype.typeof(attr)
    offset = rectype.offset(attr)

    resty = types.Array(dtype, ndim=typ.ndim, layout='A')

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


#-------------------------------------------------------------------------------
# builtin `numpy.flat` implementation

@struct_factory(types.NumpyFlatType)
def make_array_flat_cls(flatiterty):
    """
    Return the Structure representation of the given *flatiterty* (an
    instance of types.NumpyFlatType).
    """
    return _make_flattening_iter_cls(flatiterty, 'flat')


@struct_factory(types.NumpyNdEnumerateType)
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


@struct_factory(types.NumpyNdIndexType)
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
                    value = context.unpack_value(builder, arrty.dtype, ptr)
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
                return builder.load(ptr)

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
                value = context.unpack_value(builder, arrty.dtype, ptr)
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

            def getitem(self, context, builder, arrty, arr, index):
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
                return builder.load(ptr)

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
    ary = make_array(arytype)(context, builder, value=args[0])
    shapes = cgutils.unpack_tuple(builder, ary.shape, count=arytype.ndim)
    return sig.return_type, shapes


@builtin
@implement(numpy.empty, types.Any)
@implement(numpy.empty, types.Any, types.Any)
def numpy_empty_nd(context, builder, sig, args):
    arrtype, shapes = _parse_empty_args(context, builder, sig, args)
    ary = _empty_nd_impl(context, builder, arrtype, shapes)
    return impl_ret_new_ref(context, builder, sig.return_type, ary._getvalue())

@builtin
@implement(numpy.empty_like, types.Kind(types.Array))
@implement(numpy.empty_like, types.Kind(types.Array), types.Kind(types.DTypeSpec))
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
@implement(numpy.zeros_like, types.Kind(types.Array))
@implement(numpy.zeros_like, types.Kind(types.Array), types.Kind(types.DTypeSpec))
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
    @implement(numpy.full_like, types.Kind(types.Array), types.Any)
    def numpy_full_like_nd(context, builder, sig, args):

        def full_like(arr, value):
            arr = numpy.empty_like(arr)
            for idx in numpy.ndindex(arr.shape):
                arr[idx] = value
            return arr

        res = context.compile_internal(builder, full_like, sig, args)
        return impl_ret_new_ref(context, builder, sig.return_type, res)


    @builtin
    @implement(numpy.full_like, types.Kind(types.Array), types.Any, types.Kind(types.DTypeSpec))
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
@implement(numpy.ones_like, types.Kind(types.Array))
def numpy_ones_like_nd(context, builder, sig, args):

    def ones_like(arr):
        arr = numpy.empty_like(arr)
        for idx in numpy.ndindex(arr.shape):
            arr[idx] = 1
        return arr

    res = context.compile_internal(builder, ones_like, sig, args)
    return impl_ret_new_ref(context, builder, sig.return_type, res)

@builtin
@implement(numpy.ones_like, types.Kind(types.Array), types.Kind(types.DTypeSpec))
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

