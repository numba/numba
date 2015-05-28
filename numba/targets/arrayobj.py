"""
Implementation of operations on Array objects and objects supporting
the buffer protocol.
"""

from __future__ import print_function, absolute_import, division

from functools import reduce
import math

import llvmlite.llvmpy.core as lc

import numba.ctypes_support as ctypes
import numpy
from llvmlite.llvmpy.core import Constant
from numba import types, cgutils
from numba.numpy_support import as_dtype
from numba.numpy_support import version as numpy_version
from numba.targets.imputils import (builtin, builtin_attr, implement,
                                    impl_attribute, impl_attribute_generic,
                                    iterator_impl, iternext_impl,
                                    struct_factory)
from .builtins import Slice


def make_array(array_type):
    """
    Return the Structure representation of the given *array_type*
    (an instance of types.Array).
    """
    return cgutils.create_struct_proxy(array_type)


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
        nitems = context.get_constant(types.intp, 0)
    attrs['nitems'] = nitems

    # Make sure that we have all the fields
    got_fields = set(attrs.keys())
    if got_fields != required_fields:
        raise ValueError("missing {0}".format(required_fields - got_fields))

    # Set field value
    for k, v in attrs.items():
        setattr(array, k, v)

    return array


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

    return iterobj._getvalue()


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

    with cgutils.ifthen(builder, is_valid):
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
        result = _getitem_array1d(context, builder, aryty, adapted_ary, idx,
                                  wraparound=idxty.signed)
    elif ndim > 1:
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
    return result

@builtin
@implement('getitem', types.Kind(types.Buffer), types.slice3_type)
def getitem_array1d_slice(context, builder, sig, args):
    aryty, _ = sig.args
    if aryty.ndim != 1:
        # TODO
        raise NotImplementedError("1D indexing into %dD array" % aryty.ndim)

    ary, idx = args

    arystty = make_array(aryty)
    ary = arystty(context, builder, value=ary)

    shapes = cgutils.unpack_tuple(builder, ary.shape, aryty.ndim)

    slicestruct = Slice(context, builder, value=idx)
    cgutils.normalize_slice(builder, slicestruct, shapes[0])

    dataptr = cgutils.get_item_pointer(builder, aryty, ary,
                                       [slicestruct.start],
                                       wraparound=True)

    retstty = make_array(sig.return_type)
    retary = retstty(context, builder)

    shape = cgutils.get_range_from_slice(builder, slicestruct)
    stride = cgutils.get_strides_from_slice(builder, aryty.ndim, ary.strides,
                                            slicestruct, 0)

    populate_array(retary,
                   data=dataptr,
                   shape=cgutils.pack_array(builder, [shape]),
                   strides=cgutils.pack_array(builder, [stride]),
                   itemsize=ary.itemsize,
                   meminfo=ary.meminfo,
                   parent=ary.parent)

    return retary._getvalue()


@builtin
@implement('getitem', types.Kind(types.Buffer),
           types.Kind(types.UniTuple))
def getitem_array_unituple(context, builder, sig, args):
    aryty, idxty = sig.args
    ary, idx = args

    ndim = aryty.ndim
    arystty = make_array(aryty)
    ary = arystty(context, builder, ary)

    if idxty.dtype == types.slice3_type:
        # Slicing
        raw_slices = cgutils.unpack_tuple(builder, idx, aryty.ndim)
        slices = [Slice(context, builder, value=sl) for sl in raw_slices]
        for sl, sh in zip(slices,
                          cgutils.unpack_tuple(builder, ary.shape, ndim)):
            cgutils.normalize_slice(builder, sl, sh)
        indices = [sl.start for sl in slices]
        dataptr = cgutils.get_item_pointer(builder, aryty, ary, indices,
                                           wraparound=True)
        # Build array
        retstty = make_array(sig.return_type)
        retary = retstty(context, builder)
        shapes = [cgutils.get_range_from_slice(builder, sl)
                  for sl in slices]
        strides = [cgutils.get_strides_from_slice(builder, ndim, ary.strides,
                                                  sl, i)
                   for i, sl in enumerate(slices)]
        populate_array(retary,
                       data=dataptr,
                       shape=cgutils.pack_array(builder, shapes),
                       strides=cgutils.pack_array(builder, strides),
                       itemsize=ary.itemsize,
                       meminfo=ary.meminfo,
                       parent=ary.parent)
        return retary._getvalue()
    else:
        # Indexing
        assert isinstance(idxty.dtype, types.Integer)
        indices = cgutils.unpack_tuple(builder, idx, count=len(idxty))
        indices = [context.cast(builder, i, t, types.intp)
                   for t, i in zip(idxty, indices)]
        ptr = cgutils.get_item_pointer(builder, aryty, ary, indices,
                                       wraparound=idxty.dtype.signed)

        return context.unpack_value(builder, aryty.dtype, ptr)


@builtin
@implement('getitem', types.Kind(types.Buffer),
           types.Kind(types.Tuple))
def getitem_array_tuple(context, builder, sig, args):
    aryty, idxty = sig.args
    ary, idx = args

    arystty = make_array(aryty)
    ary = arystty(context, builder, ary)

    ndim = aryty.ndim
    if isinstance(sig.return_type, types.Array):
        # Slicing
        raw_indices = cgutils.unpack_tuple(builder, idx, aryty.ndim)
        start = []
        shapes = []
        strides = []

        oshapes = cgutils.unpack_tuple(builder, ary.shape, ndim)
        for ax, (indexval, idxty) in enumerate(zip(raw_indices, idxty)):
            if idxty == types.slice3_type:
                slice = Slice(context, builder, value=indexval)
                cgutils.normalize_slice(builder, slice, oshapes[ax])
                start.append(slice.start)
                shapes.append(cgutils.get_range_from_slice(builder, slice))
                strides.append(cgutils.get_strides_from_slice(builder, ndim,
                                                              ary.strides,
                                                              slice, ax))
            else:
                ind = context.cast(builder, indexval, idxty, types.intp)
                start.append(ind)

        dataptr = cgutils.get_item_pointer(builder, aryty, ary, start,
                                           wraparound=True)
        # Build array
        retstty = make_array(sig.return_type)
        retary = retstty(context, builder)
        populate_array(retary,
                       data=dataptr,
                       shape=cgutils.pack_array(builder, shapes),
                       strides=cgutils.pack_array(builder, strides),
                       itemsize=ary.itemsize,
                       meminfo=ary.meminfo,
                       parent=ary.parent)
        return retary._getvalue()
    else:
        # Indexing
        indices = cgutils.unpack_tuple(builder, idx, count=len(idxty))
        indices = [context.cast(builder, i, t, types.intp)
                   for t, i in zip(idxty, indices)]
        ptr = cgutils.get_item_pointer(builder, aryty, ary, indices,
                                       wraparound=True)

        return context.unpack_value(builder, aryty.dtype, ptr)


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
    slicestruct = Slice(context, builder, value=idx)

    # the logic here follows that of Python's Objects/sliceobject.c
    # in particular PySlice_GetIndicesEx function
    ZERO = Constant.int(slicestruct.step.type, 0)
    NEG_ONE = Constant.int(slicestruct.start.type, -1)

    b_step_eq_zero = builder.icmp(lc.ICMP_EQ, slicestruct.step, ZERO)
    # bail if step is 0
    with cgutils.ifthen(builder, b_step_eq_zero):
        context.call_conv.return_user_exc(builder, ValueError,
                                          ("slice step cannot be zero",))

    # adjust for negative indices for start
    start = cgutils.alloca_once_value(builder, slicestruct.start)
    b_start_lt_zero = builder.icmp(lc.ICMP_SLT, builder.load(start), ZERO)
    with cgutils.ifthen(builder, b_start_lt_zero):
        add = builder.add(builder.load(start), shapes[0])
        builder.store(add, start)

    b_start_lt_zero = builder.icmp(lc.ICMP_SLT, builder.load(start), ZERO)
    with cgutils.ifthen(builder, b_start_lt_zero):
        b_step_lt_zero = builder.icmp(lc.ICMP_SLT, slicestruct.step, ZERO)
        cond = builder.select(b_step_lt_zero, NEG_ONE, ZERO)
        builder.store(cond, start)

    b_start_geq_len = builder.icmp(lc.ICMP_SGE, builder.load(start), shapes[0])
    ONE = Constant.int(shapes[0].type, 1)
    with cgutils.ifthen(builder, b_start_geq_len):
        b_step_lt_zero = builder.icmp(lc.ICMP_SLT, slicestruct.step, ZERO)
        cond = builder.select(b_step_lt_zero, builder.sub(shapes[0], ONE), shapes[0])
        builder.store(cond, start)

    # adjust stop for negative value
    stop = cgutils.alloca_once_value(builder, slicestruct.stop)
    b_stop_lt_zero = builder.icmp(lc.ICMP_SLT, builder.load(stop), ZERO)
    with cgutils.ifthen(builder, b_stop_lt_zero):
        add = builder.add(builder.load(stop), shapes[0])
        builder.store(add, stop)

    b_stop_lt_zero = builder.icmp(lc.ICMP_SLT, builder.load(stop), ZERO)
    with cgutils.ifthen(builder, b_stop_lt_zero):
        b_step_lt_zero = builder.icmp(lc.ICMP_SLT, slicestruct.step, ZERO)
        cond = builder.select(b_step_lt_zero, NEG_ONE, ZERO)
        builder.store(cond, start)

    b_stop_geq_len = builder.icmp(lc.ICMP_SGE, builder.load(stop), shapes[0])
    ONE = Constant.int(shapes[0].type, 1)
    with cgutils.ifthen(builder, b_stop_geq_len):
        b_step_lt_zero = builder.icmp(lc.ICMP_SLT, slicestruct.step, ZERO)
        cond = builder.select(b_step_lt_zero, builder.sub(shapes[0], ONE), shapes[0])
        builder.store(cond, stop)

    b_step_gt_zero = builder.icmp(lc.ICMP_SGT, slicestruct.step, ZERO)
    with cgutils.ifelse(builder, b_step_gt_zero) as (then0, otherwise0):
        with then0:
            with cgutils.for_range_slice(builder, builder.load(start), builder.load(stop), slicestruct.step, slicestruct.start.type) as loop_idx1:
                ptr = cgutils.get_item_pointer(builder, aryty, ary,
                                   [loop_idx1],
                                   wraparound=True)
                context.pack_value(builder, aryty.dtype, val, ptr)
        with otherwise0:
            with cgutils.for_range_slice(builder, builder.load(start), builder.load(stop), slicestruct.step, slicestruct.start.type, inc=False) as loop_idx2:
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
    return builder.extract_value(shapeary, 0)


@builtin
@implement(numpy.sum, types.Kind(types.Array))
@implement("array.sum", types.Kind(types.Array))
def array_sum(context, builder, sig, args):
    [arrty] = sig.args

    def array_sum_impl(arr):
        c = 0
        for v in arr.flat:
            c += v
        return c

    return context.compile_internal(builder, array_sum_impl, sig, args,
                                    locals=dict(c=sig.return_type))


@builtin
@implement(numpy.prod, types.Kind(types.Array))
@implement("array.prod", types.Kind(types.Array))
def array_prod(context, builder, sig, args):
    [arrty] = sig.args

    def array_prod_impl(arr):
        c = 1
        for v in arr.flat:
            c *= v
        return c

    return context.compile_internal(builder, array_prod_impl, sig, args,
                                    locals=dict(c=sig.return_type))


@builtin
@implement(numpy.cumsum, types.Kind(types.Array))
@implement("array.cumsum", types.Kind(types.Array))
def array_cumsum(context, builder, sig, args):
    scalar_dtype = sig.return_type.dtype
    dtype = getattr(numpy, str(scalar_dtype))

    def array_cumsum_impl(arr):
        size = 1
        for i in arr.shape:
            size = size * i
        out = numpy.empty(size, dtype)
        c = 0
        for idx, v in enumerate(arr.flat):
            c += v
            out[idx] = c
        return out

    return context.compile_internal(builder, array_cumsum_impl, sig, args,
                                    locals=dict(c=scalar_dtype))


@builtin
@implement(numpy.cumprod, types.Kind(types.Array))
@implement("array.cumprod", types.Kind(types.Array))
def array_cumsum(context, builder, sig, args):
    scalar_dtype = sig.return_type.dtype
    dtype = getattr(numpy, str(scalar_dtype))

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

    return context.compile_internal(builder, array_cumprod_impl, sig, args,
                                    locals=dict(c=scalar_dtype))


@builtin
@implement(numpy.mean, types.Kind(types.Array))
@implement("array.mean", types.Kind(types.Array))
def array_mean(context, builder, sig, args):
    [arrty] = sig.args

    def array_mean_impl(arr):
        # Can't use the naive `arr.sum() / arr.size`, as it would return
        # a wrong result on integer sum overflow.
        c = 0
        for v in arr.flat:
            c += v
        return c / arr.size

    return context.compile_internal(builder, array_mean_impl, sig, args,
                                    locals=dict(c=sig.return_type))


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

    return context.compile_internal(builder, array_var_impl, sig, args)


@builtin
@implement(numpy.std, types.Kind(types.Array))
@implement("array.std", types.Kind(types.Array))
def array_std(context, builder, sig, args):
    def array_std_impl(arry):
        return arry.var() ** 0.5
    return context.compile_internal(builder, array_std_impl, sig, args)


@builtin
@implement(numpy.min, types.Kind(types.Array))
@implement("array.min", types.Kind(types.Array))
def array_min(context, builder, sig, args):
    def array_min_impl(arry):
        for v in arry.flat:
            min_value = v
            break

        for v in arry.flat:
            if v < min_value:
                min_value = v
        return min_value
    return context.compile_internal(builder, array_min_impl, sig, args)


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
    return context.compile_internal(builder, array_max_impl, sig, args)


@builtin
@implement(numpy.argmin, types.Kind(types.Array))
@implement("array.argmin", types.Kind(types.Array))
def array_argmin(context, builder, sig, args):
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
    return context.compile_internal(builder, array_argmin_impl, sig, args)


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
    return context.compile_internal(builder, array_argmax_impl, sig, args)


def _np_round_intrinsic(tp):
    # np.round() always rounds half to even
    return "llvm.rint.f%d" % (tp.bitwidth,)

def _np_round_float(context, builder, tp, val):
    llty = context.get_value_type(tp)
    module = cgutils.get_module(builder)
    fnty = lc.Type.function(llty, [llty])
    fn = module.get_or_insert_function(fnty, name=_np_round_intrinsic(tp))
    return builder.call(fn, (val,))

@builtin
@implement(numpy.round, types.Kind(types.Float))
def scalar_round_unary(context, builder, sig, args):
    return _np_round_float(context, builder, sig.args[0], args[0])

@builtin
@implement(numpy.round, types.Kind(types.Integer))
def scalar_round_unary(context, builder, sig, args):
    return args[0]

@builtin
@implement(numpy.round, types.Kind(types.Complex))
def scalar_round_unary_complex(context, builder, sig, args):
    fltty = sig.args[0].underlying_float
    cplx_cls = context.make_complex(sig.args[0])
    z = cplx_cls(context, builder, args[0])
    z.real = _np_round_float(context, builder, fltty, z.real)
    z.imag = _np_round_float(context, builder, fltty, z.imag)
    return z._getvalue()

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

    return context.compile_internal(builder, round_ndigits, sig, args)

@builtin
@implement(numpy.round, types.Kind(types.Complex), types.Kind(types.Integer))
def scalar_round_binary_complex(context, builder, sig, args):
    def round_ndigits(z, ndigits):
        return complex(numpy.round(z.real, ndigits),
                       numpy.round(z.imag, ndigits))

    return context.compile_internal(builder, round_ndigits, sig, args)


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

    return context.compile_internal(builder, array_round_impl, sig, args)


#-------------------------------------------------------------------------------


@builtin_attr
@impl_attribute(types.Kind(types.Array), "shape", types.Kind(types.UniTuple))
@impl_attribute(types.Kind(types.MemoryView), "shape", types.Kind(types.UniTuple))
def array_shape(context, builder, typ, value):
    arrayty = make_array(typ)
    array = arrayty(context, builder, value)
    return array.shape


@builtin_attr
@impl_attribute(types.Kind(types.Array), "strides", types.Kind(types.UniTuple))
@impl_attribute(types.Kind(types.MemoryView), "strides", types.Kind(types.UniTuple))
def array_strides(context, builder, typ, value):
    arrayty = make_array(typ)
    array = arrayty(context, builder, value)
    return array.strides


@builtin_attr
@impl_attribute(types.Kind(types.Array), "ndim", types.intp)
@impl_attribute(types.Kind(types.MemoryView), "ndim", types.intp)
def array_ndim(context, builder, typ, value):
    return context.get_constant(types.intp, typ.ndim)


@builtin_attr
@impl_attribute(types.Kind(types.Array), "size", types.intp)
def array_size(context, builder, typ, value):
    arrayty = make_array(typ)
    array = arrayty(context, builder, value)
    return array.nitems


@builtin_attr
@impl_attribute(types.Kind(types.Array), "itemsize", types.intp)
@impl_attribute(types.Kind(types.MemoryView), "itemsize", types.intp)
def array_itemsize(context, builder, typ, value):
    arrayty = make_array(typ)
    array = arrayty(context, builder, value)
    return array.itemsize


@builtin_attr
@impl_attribute(types.Kind(types.MemoryView), "nbytes", types.intp)
def array_nbytes(context, builder, typ, value):
    """
    nbytes = size * itemsize
    """
    arrayty = make_array(typ)
    array = arrayty(context, builder, value)
    dims = cgutils.unpack_tuple(builder, array.shape, typ.ndim)
    return builder.mul(array.nitems, array.itemsize)


@builtin_attr
@impl_attribute(types.Kind(types.MemoryView), "contiguous", types.boolean)
def array_contiguous(context, builder, typ, value):
    return context.get_constant(types.boolean, typ.is_contig)

@builtin_attr
@impl_attribute(types.Kind(types.MemoryView), "c_contiguous", types.boolean)
def array_c_contiguous(context, builder, typ, value):
    return context.get_constant(types.boolean, typ.is_c_contig)

@builtin_attr
@impl_attribute(types.Kind(types.MemoryView), "f_contiguous", types.boolean)
def array_f_contiguous(context, builder, typ, value):
    return context.get_constant(types.boolean, typ.is_f_contig)


@builtin_attr
@impl_attribute(types.Kind(types.MemoryView), "readonly", types.boolean)
def array_readonly(context, builder, typ, value):
    return context.get_constant(types.boolean, not typ.mutable)


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
    return ctinfo._getvalue()


@builtin_attr
@impl_attribute(types.Kind(types.ArrayCTypes), "data", types.uintp)
def array_ctypes_data(context, builder, typ, value):
    ctinfo_type = cgutils.create_struct_proxy(typ)
    ctinfo = ctinfo_type(context, builder, value=value)
    return ctinfo.data


@builtin_attr
@impl_attribute_generic(types.Kind(types.Array))
def array_record_getattr(context, builder, typ, value, attr):
    arrayty = make_array(typ)
    array = arrayty(context, builder, value)

    rectype = typ.dtype
    assert isinstance(rectype, types.Record)
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
    return rary._getvalue()


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

    bbend = cgutils.append_basic_block(builder, 'end_increment')

    if end_flag is not None:
        builder.store(cgutils.false_byte, end_flag)

    for dim in reversed(range(ndim)):
        idxptr = cgutils.gep(builder, indices, dim)
        idx = builder.add(builder.load(idxptr), one)

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
                idxptr = cgutils.gep(builder, indices, dim)
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
            self.shape = cgutils.pack_array(builder, shapes)

        def iternext_specific(self, context, builder, result):
            zero = context.get_constant(types.intp, 0)
            one = context.get_constant(types.intp, 1)

            bbend = cgutils.append_basic_block(builder, 'end')

            exhausted = cgutils.as_bool_bit(builder, builder.load(self.exhausted))
            with cgutils.if_unlikely(builder, exhausted):
                result.set_valid(False)
                builder.branch(bbend)

            indices = [builder.load(cgutils.gep(builder, self.indices, dim))
                       for dim in range(ndim)]
            result.yield_(cgutils.pack_array(builder, indices))
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
                self.pointer = cgutils.alloca_once_value(builder, arr.data)
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
                        idxptr = cgutils.gep(builder, indices, dim)
                        builder.store(zero, idxptr)

                    self.indices = indices

            def iternext_specific(self, context, builder, arrty, arr, result):
                zero = context.get_constant(types.intp, 0)
                one = context.get_constant(types.intp, 1)

                ndim = arrty.ndim
                nitems = arr.nitems

                index = builder.load(self.index)
                is_valid = builder.icmp(lc.ICMP_SLT, index, nitems)
                result.set_valid(is_valid)

                with cgutils.if_likely(builder, is_valid):
                    ptr = builder.load(self.pointer)
                    value = context.unpack_value(builder, arrty.dtype, ptr)
                    if kind == 'flat':
                        result.yield_(value)
                    else:
                        # ndenumerate(): fetch and increment indices
                        indices = self.indices
                        idxvals = [builder.load(cgutils.gep(builder, indices, dim))
                                   for dim in range(ndim)]
                        idxtuple = cgutils.pack_array(builder, idxvals)
                        result.yield_(
                            cgutils.make_anonymous_struct(builder, [idxtuple, value]))
                        _increment_indices_array(context, builder, arrty, arr, indices)

                    index = builder.add(index, one)
                    builder.store(index, self.index)
                    ptr = cgutils.pointer_add(builder, ptr, self.stride)
                    builder.store(ptr, self.pointer)

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
                    idxptr = cgutils.gep(builder, indices, dim)
                    ptrptr = cgutils.gep(builder, pointers, dim)
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

                bbend = cgutils.append_basic_block(builder, 'end')

                # Catch already computed iterator exhaustion
                is_exhausted = cgutils.as_bool_bit(
                    builder, builder.load(self.exhausted))
                with cgutils.if_unlikely(builder, is_exhausted):
                    result.set_valid(False)
                    builder.branch(bbend)
                result.set_valid(True)

                # Current pointer inside last dimension
                last_ptr = cgutils.gep(builder, pointers, ndim - 1)
                ptr = builder.load(last_ptr)
                value = context.unpack_value(builder, arrty.dtype, ptr)
                if kind == 'flat':
                    result.yield_(value)
                else:
                    # ndenumerate() => yield (indices, value)
                    idxvals = [builder.load(cgutils.gep(builder, indices, dim))
                               for dim in range(ndim)]
                    idxtuple = cgutils.pack_array(builder, idxvals)
                    result.yield_(
                        cgutils.make_anonymous_struct(builder, [idxtuple, value]))

                # Update indices and pointers by walking from inner
                # dimension to outer.
                for dim in reversed(range(ndim)):
                    idxptr = cgutils.gep(builder, indices, dim)
                    idx = builder.add(builder.load(idxptr), one)

                    count = shapes[dim]
                    stride = strides[dim]
                    in_bounds = builder.icmp(lc.ICMP_SLT, idx, count)
                    with cgutils.if_likely(builder, in_bounds):
                        # Index is valid => pointer can simply be incremented.
                        builder.store(idx, idxptr)
                        ptrptr = cgutils.gep(builder, pointers, dim)
                        ptr = builder.load(ptrptr)
                        ptr = cgutils.pointer_add(builder, ptr, stride)
                        builder.store(ptr, ptrptr)
                        # Reset pointers in inner dimensions
                        for inner_dim in range(dim + 1, ndim):
                            ptrptr = cgutils.gep(builder, pointers, inner_dim)
                            builder.store(ptr, ptrptr)
                        builder.branch(bbend)
                    # Reset index and continue with next dimension
                    builder.store(zero, idxptr)

                # End of array
                builder.store(cgutils.true_byte, self.exhausted)
                builder.branch(bbend)

                builder.position_at_end(bbend)

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

    return flatiter._getvalue()


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

    return nditer._getvalue()


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

    return nditer._getvalue()

@builtin
@implement(numpy.ndindex, types.Kind(types.UniTuple))
def make_array_ndindex(context, builder, sig, args):
    """ndindex(shape)"""
    ndim = sig.return_type.ndim
    idxty = sig.args[0].dtype
    tup = args[0]

    shape = cgutils.unpack_tuple(builder, tup, ndim)
    shape = [context.cast(builder, idx, idxty, types.intp)
             for idx in shape]

    nditercls = make_ndindex_cls(types.NumpyNdIndexType(len(shape)))
    nditer = nditercls(context, builder)
    nditer.init_specific(context, builder, shape)

    return nditer._getvalue()

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

    if arrtype.layout == 'C':
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

    meminfo = context.nrt_meminfo_alloc(builder,
                                        size=builder.mul(itemsize, arrlen))
    data = context.nrt_meminfo_data(builder, meminfo)

    populate_array(ary,
                   data=builder.bitcast(data, datatype.as_pointer()),
                   shape=cgutils.pack_array(builder, shapes),
                   strides=cgutils.pack_array(builder, strides),
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
    return ary._getvalue()

@builtin
@implement(numpy.empty_like, types.Kind(types.Array))
@implement(numpy.empty_like, types.Kind(types.Array), types.Kind(types.DTypeSpec))
def numpy_empty_like_nd(context, builder, sig, args):
    arrtype, shapes = _parse_empty_like_args(context, builder, sig, args)
    ary = _empty_nd_impl(context, builder, arrtype, shapes)
    return ary._getvalue()


@builtin
@implement(numpy.zeros, types.Any)
@implement(numpy.zeros, types.Any, types.Any)
def numpy_zeros_nd(context, builder, sig, args):
    arrtype, shapes = _parse_empty_args(context, builder, sig, args)
    ary = _empty_nd_impl(context, builder, arrtype, shapes)
    _zero_fill_array(context, builder, ary)
    return ary._getvalue()


@builtin
@implement(numpy.zeros_like, types.Kind(types.Array))
@implement(numpy.zeros_like, types.Kind(types.Array), types.Kind(types.DTypeSpec))
def numpy_zeros_like_nd(context, builder, sig, args):
    arrtype, shapes = _parse_empty_like_args(context, builder, sig, args)
    ary = _empty_nd_impl(context, builder, arrtype, shapes)
    _zero_fill_array(context, builder, ary)
    return ary._getvalue()


if numpy_version >= (1, 8):
    @builtin
    @implement(numpy.full, types.Any, types.Any)
    def numpy_full_nd(context, builder, sig, args):

        def full(shape, value):
            arr = numpy.empty(shape)
            for idx in numpy.ndindex(arr.shape):
                arr[idx] = value
            return arr

        return context.compile_internal(builder, full, sig, args)

    @builtin
    @implement(numpy.full, types.Any, types.Any, types.Kind(types.DTypeSpec))
    def numpy_full_dtype_nd(context, builder, sig, args):

        def full(shape, value, dtype):
            arr = numpy.empty(shape, dtype)
            for idx in numpy.ndindex(arr.shape):
                arr[idx] = value
            return arr

        return context.compile_internal(builder, full, sig, args)


    @builtin
    @implement(numpy.full_like, types.Kind(types.Array), types.Any)
    def numpy_full_like_nd(context, builder, sig, args):

        def full_like(arr, value):
            arr = numpy.empty_like(arr)
            for idx in numpy.ndindex(arr.shape):
                arr[idx] = value
            return arr

        return context.compile_internal(builder, full_like, sig, args)


    @builtin
    @implement(numpy.full_like, types.Kind(types.Array), types.Any, types.Kind(types.DTypeSpec))
    def numpy_full_like_nd(context, builder, sig, args):

        def full_like(arr, value, dtype):
            arr = numpy.empty_like(arr, dtype)
            for idx in numpy.ndindex(arr.shape):
                arr[idx] = value
            return arr

        return context.compile_internal(builder, full_like, sig, args)


@builtin
@implement(numpy.ones, types.Any)
def numpy_ones_nd(context, builder, sig, args):

    def ones(shape):
        arr = numpy.empty(shape)
        for idx in numpy.ndindex(arr.shape):
            arr[idx] = 1
        return arr

    valty = sig.return_type.dtype
    return context.compile_internal(builder, ones, sig, args,
                                    locals={'c': valty})

@builtin
@implement(numpy.ones, types.Any, types.Kind(types.DTypeSpec))
def numpy_ones_dtype_nd(context, builder, sig, args):

    def ones(shape, dtype):
        arr = numpy.empty(shape, dtype)
        for idx in numpy.ndindex(arr.shape):
            arr[idx] = 1
        return arr

    return context.compile_internal(builder, ones, sig, args)

@builtin
@implement(numpy.ones_like, types.Kind(types.Array))
def numpy_ones_like_nd(context, builder, sig, args):

    def ones_like(arr):
        arr = numpy.empty_like(arr)
        for idx in numpy.ndindex(arr.shape):
            arr[idx] = 1
        return arr

    return context.compile_internal(builder, ones_like, sig, args)

@builtin
@implement(numpy.ones_like, types.Kind(types.Array), types.Kind(types.DTypeSpec))
def numpy_ones_like_dtype_nd(context, builder, sig, args):

    def ones_like(arr, dtype):
        arr = numpy.empty_like(arr, dtype)
        for idx in numpy.ndindex(arr.shape):
            arr[idx] = 1
        return arr

    return context.compile_internal(builder, ones_like, sig, args)


@builtin
@implement(numpy.identity, types.Kind(types.Integer))
def numpy_identity(context, builder, sig, args):

    def identity(n):
        arr = numpy.zeros((n, n))
        for i in range(n):
            arr[i, i] = 1
        return arr

    return context.compile_internal(builder, identity, sig, args)

@builtin
@implement(numpy.identity, types.Kind(types.Integer), types.Kind(types.DTypeSpec))
def numpy_identity(context, builder, sig, args):

    def identity(n, dtype):
        arr = numpy.zeros((n, n), dtype)
        for i in range(n):
            arr[i, i] = 1
        return arr

    return context.compile_internal(builder, identity, sig, args)


@builtin
@implement(numpy.eye, types.Kind(types.Integer))
def numpy_eye(context, builder, sig, args):

    def eye(n):
        return numpy.identity(n)

    return context.compile_internal(builder, eye, sig, args)

@builtin
@implement(numpy.eye, types.Kind(types.Integer), types.Kind(types.Integer))
def numpy_eye(context, builder, sig, args):

    def eye(n, m):
        return numpy.eye(n, m, 0, numpy.float64)

    return context.compile_internal(builder, eye, sig, args)

@builtin
@implement(numpy.eye, types.Kind(types.Integer), types.Kind(types.Integer),
           types.Kind(types.Integer))
def numpy_eye(context, builder, sig, args):

    def eye(n, m, k):
        return numpy.eye(n, m, k, numpy.float64)

    return context.compile_internal(builder, eye, sig, args)

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

    return context.compile_internal(builder, eye, sig, args)


@builtin
@implement(numpy.arange, types.Kind(types.Number))
def numpy_arange_1(context, builder, sig, args):
    dtype = getattr(numpy, str(sig.return_type.dtype))

    def arange(stop):
        return numpy.arange(0, stop, 1, dtype)

    return context.compile_internal(builder, arange, sig, args)

@builtin
@implement(numpy.arange, types.Kind(types.Number), types.Kind(types.Number))
def numpy_arange_2(context, builder, sig, args):
    dtype = getattr(numpy, str(sig.return_type.dtype))

    def arange(start, stop):
        return numpy.arange(start, stop, 1, dtype)

    return context.compile_internal(builder, arange, sig, args)


@builtin
@implement(numpy.arange, types.Kind(types.Number), types.Kind(types.Number),
           types.Kind(types.Number))
def numpy_arange_3(context, builder, sig, args):
    dtype = getattr(numpy, str(sig.return_type.dtype))

    def arange(start, stop, step):
        return numpy.arange(start, stop, step, dtype)

    return context.compile_internal(builder, arange, sig, args)

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

    return context.compile_internal(builder, arange, sig, args,
                                    locals={'nitems': types.intp})


@builtin
@implement(numpy.linspace, types.Kind(types.Number), types.Kind(types.Number))
def numpy_linspace_2(context, builder, sig, args):

    def linspace(start, stop):
        return numpy.linspace(start, stop, 50)

    return context.compile_internal(builder, linspace, sig, args)

@builtin
@implement(numpy.linspace, types.Kind(types.Number), types.Kind(types.Number),
           types.Kind(types.Integer))
def numpy_linspace_3(context, builder, sig, args):
    dtype = getattr(numpy, str(sig.return_type.dtype))

    def linspace(start, stop, num):
        arr = numpy.empty(num, dtype)
        div = num - 1
        delta = stop - start
        arr[0] = start
        for i in range(1, num):
            arr[i] = start + delta * (i / div)
        return arr

    return context.compile_internal(builder, linspace, sig, args)

