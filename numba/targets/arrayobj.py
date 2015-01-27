"""
Implementation of operations on Array objects.
"""

from __future__ import print_function, absolute_import, division

from functools import reduce

import llvmlite.llvmpy.core as lc

import numba.ctypes_support as ctypes
import numpy
from llvmlite.llvmpy.core import Constant
from numba import errcode
from numba import types, cgutils
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
    dtype = array_type.dtype
    nd = array_type.ndim

    # This structure should be kept in sync with Numba_adapt_ndarray()
    # in _helperlib.c.
    class ArrayTemplate(cgutils.Structure):
        _fields = [('parent', types.pyobject),
                   ('nitems', types.intp),
                   ('itemsize', types.intp),
                   # These three fields comprise the unofficiel llarray ABI
                   # (used by the GPU backend)
                   ('data', types.CPointer(dtype)),
                   ('shape', types.UniTuple(types.intp, nd)),
                   ('strides', types.UniTuple(types.intp, nd)),
                   ]

    return ArrayTemplate

def make_array_ctype(ndim):
    """Create a ctypes representation of an array_type.

    Parameters
    -----------
    ndim: int
        number of dimensions of array

    Returns
    -----------
        a ctypes array structure for an array with the given number of
        dimensions
    """
    c_intp = ctypes.c_ssize_t

    class c_array(ctypes.Structure):
        _fields_ = [('parent', ctypes.c_void_p),
                    ('nitems', c_intp),
                    ('itemsize', c_intp),
                    ('data', ctypes.c_void_p),
                    ('shape', c_intp * ndim),
                    ('strides', c_intp * ndim)]

    return c_array


@struct_factory(types.ArrayIterator)
def make_arrayiter_cls(iterator_type):
    """
    Return the Structure representation of the given *iterator_type* (an
    instance of types.ArrayIteratorType).
    """

    class ArrayIteratorStruct(cgutils.Structure):
        # We use an unsigned index to avoid the cost of negative index tests.
        _fields = [('index', types.CPointer(types.uintp)),
                   ('array', iterator_type.array_type)]

    return ArrayIteratorStruct

@builtin
@implement('getiter', types.Kind(types.Array))
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
@implement('getitem', types.Kind(types.Array), types.Kind(types.Integer))
def getitem_array1d_intp(context, builder, sig, args):
    aryty, idxty = sig.args
    if aryty.ndim != 1:
        # TODO
        raise NotImplementedError("1D indexing into %dD array" % aryty.ndim)

    ary, idx = args

    arystty = make_array(aryty)
    ary = arystty(context, builder, ary)
    return _getitem_array1d(context, builder, aryty, ary, idx,
                            wraparound=idxty.signed)

@builtin
@implement('getitem', types.Kind(types.Array), types.slice3_type)
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
    retary.shape = cgutils.pack_array(builder, [shape])

    stride = cgutils.get_strides_from_slice(builder, aryty.ndim, ary.strides,
                                            slicestruct, 0)
    retary.strides = cgutils.pack_array(builder, [stride])
    retary.data = dataptr

    return retary._getvalue()


@builtin
@implement('getitem', types.Kind(types.Array),
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
        retary.data = dataptr
        shapes = [cgutils.get_range_from_slice(builder, sl)
                  for sl in slices]
        retary.shape = cgutils.pack_array(builder, shapes)
        strides = [cgutils.get_strides_from_slice(builder, ndim, ary.strides,
                                                  sl, i)
                   for i, sl in enumerate(slices)]

        retary.strides = cgutils.pack_array(builder, strides)

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
@implement('getitem', types.Kind(types.Array),
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
        retary.data = dataptr
        retary.shape = cgutils.pack_array(builder, shapes)
        retary.strides = cgutils.pack_array(builder, strides)
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
@implement('setitem', types.Kind(types.Array), types.Kind(types.Integer),
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
@implement('setitem', types.Kind(types.Array),
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
@implement('setitem', types.Kind(types.Array),
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
@implement('setitem', types.Kind(types.Array),
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
        context.return_errcode(builder, errcode.ASSERTION_ERROR)

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
@implement(types.len_type, types.Kind(types.Array))
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


#-------------------------------------------------------------------------------


@builtin_attr
@impl_attribute(types.Kind(types.Array), "shape", types.Kind(types.UniTuple))
def array_shape(context, builder, typ, value):
    arrayty = make_array(typ)
    array = arrayty(context, builder, value)
    return array.shape


@builtin_attr
@impl_attribute(types.Kind(types.Array), "strides", types.Kind(types.UniTuple))
def array_strides(context, builder, typ, value):
    arrayty = make_array(typ)
    array = arrayty(context, builder, value)
    return array.strides


@builtin_attr
@impl_attribute(types.Kind(types.Array), "ndim", types.intp)
def array_ndim(context, builder, typ, value):
    return context.get_constant(types.intp, typ.ndim)


@builtin_attr
@impl_attribute(types.Kind(types.Array), "size", types.intp)
def array_size(context, builder, typ, value):
    arrayty = make_array(typ)
    array = arrayty(context, builder, value)
    dims = cgutils.unpack_tuple(builder, array.shape, typ.ndim)
    return reduce(builder.mul, dims[1:], dims[0])


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
    rary.shape = array.shape

    constoffset = context.get_constant(types.intp, offset)
    unpackedstrides = cgutils.unpack_tuple(builder, array.strides, typ.ndim)
    newstrides = [builder.add(s, constoffset) for s in unpackedstrides]

    rary.strides = array.strides

    llintp = context.get_value_type(types.intp)
    newdata = builder.add(builder.ptrtoint(array.data, llintp), constoffset)
    newdataptr = builder.inttoptr(newdata, rary.data.type)
    rary.data = newdataptr

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

    class NdIndexIter(cgutils.Structure):
        """
        .ndindex() implementation.
        """
        _fields = [('shape', types.UniTuple(types.intp, ndim)),
                   ('indices', types.CPointer(types.intp)),
                   ('exhausted', types.CPointer(types.boolean)),
                   ]

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
        class CContiguousFlatIter(cgutils.Structure):
            """
            .flat() / .ndenumerate() implementation for C-contiguous arrays.
            """
            _fields = [('array', types.CPointer(array_type)),
                       ('stride', types.intp),
                       ('pointer', types.CPointer(types.CPointer(dtype))),
                       ('index', types.CPointer(types.intp)),
                       ('indices', types.CPointer(types.intp)),
                       ]

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
        class FlatIter(cgutils.Structure):
            """
            Generic .flat() / .ndenumerate() implementation for
            non-contiguous arrays.
            It keeps track of pointers along each dimension in order to
            minimize computations.
            """
            _fields = [('array', types.CPointer(array_type)),
                       ('pointers', types.CPointer(types.CPointer(dtype))),
                       ('indices', types.CPointer(types.intp)),
                       ('exhausted', types.CPointer(types.boolean)),
                       ]

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
