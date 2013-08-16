'''
Implements numpy array functions.
'''
import operator
import numpy
from . import aryutils, types

def imp_numpy_sum(context, args, argtys, retty):
    '''
    tmp = 0
    for i in elementwise(ary):
        tmp += i
    return tmp
    '''
    builder = context.builder
    (ary,) = args
    (aryty,) = argtys
    elemty = aryty.desc.element

    sum = builder.alloca(elemty.llvm_as_value())
    builder.store(elemty.llvm_const(0), sum)
    with aryutils.elementwise(builder, ary, aryty) as val:
        # XXX: move to next level and reuse NPM to compile this
        assert isinstance(elemty.desc, types.Float)
        do_add = context.imp.lookup(operator.add, (elemty, elemty))
        new_sum = do_add(context, (context.builder.load(sum), val))
        builder.store(new_sum, sum)
    return builder.load(sum)

def imp_numpy_prod(context, args, argtys, retty):
    '''
    tmp = 0
    for i in elementwise(ary):
        tmp *= i
    return tmp
    '''
    builder = context.builder
    (ary,) = args
    (aryty,) = argtys
    elemty = aryty.desc.element

    prod = builder.alloca(elemty.llvm_as_value())
    builder.store(elemty.llvm_const(1), prod)
    with aryutils.elementwise(builder, ary, aryty) as val:
        # XXX: move to next level and reuse NPM to compile this
        assert isinstance(elemty.desc, types.Float)
        do_mul = context.imp.lookup(operator.mul, (elemty, elemty))
        new_prod = do_mul(context, (context.builder.load(prod), val))
        builder.store(new_prod, prod)
    return builder.load(prod)

def getitem_fixedarray_intp(args):
    ary = args[0]
    return types.tupletype(*[types.intp] * ary.desc.ndim)

def getitem_fixedarray_slice2(args):
    ary = args[0]
    return types.tupletype(*[types.slice2] * ary.desc.ndim)

def getitem_tuple(args):
    ary, idx = args
    if isinstance(idx.desc, types.Tuple):
        def either_int_or_slice(x):
            return isinstance(x.desc, (types.Integer, types.Slice2))
        if all(either_int_or_slice(x) for x in idx.desc.elements):
            return idx
    return 

def array_complex_slicing_return(args):
    ary, idx = args
    ct = 0
    for x in idx.desc.elements:
        if isinstance(x.desc, types.Integer):
            ct += 1
    ndim = ary.desc.ndim - ct
    assert ndim > 0
    return ary.desc.copy(ndim=ndim, order='A')

def array_dtype_return(args):
    return args[0].desc.element

def intp_tuple_return(args):
    ary = args[0]
    return types.tupletype(*([types.intp] * ary.desc.ndim))

def array_setitem_value(args):
    ary = args[0]
    return ary.desc.element

def array_slice_return(args):
    ary = args[0]
    if ary.desc.order == 'A':
        return ary
    else:
        return ary.desc.copy(order='%sS' % ary.desc.order[0])

def boundcheck(context, ary, indices):
    assert isinstance(indices, (tuple, list))
    if context.flags.boundcheck:
        aryutils.boundcheck(context.builder, context.raises, ary, indices)

def wraparound(context, ary, indices):
    assert isinstance(indices, (tuple, list))
    if context.flags.wraparound:
        return aryutils.wraparound(context.builder, ary, indices)
    else:
        return indices

def clip(context, ary, indices, plus1=False):
    assert isinstance(indices, (tuple, list))
    return aryutils.clip(context.builder, ary, indices, plus1=plus1)

def wraparound_and_boundcheck(context, ary, indices):
    indices = wraparound(context, ary, indices)
    boundcheck(context, ary, indices)
    return indices

def wraparound_and_clip(context, ary, indices, plus1=False):
    indices = wraparound(context, ary, indices)
    indices = clip(context, ary, indices, plus1=plus1)
    return indices

def axis_wraparound_and_clip(context, val, end, plus1=False):
    builder = context.builder
    if context.flags.wraparound:
        val = aryutils.axis_wraparound(builder, val, end)
    if plus1:
        val = aryutils.axis_clip(builder, val, end)
    else:
        endm1 = builder.sub(end, types.auto_intp(1))
        val = aryutils.axis_clip(builder, val, endm1)
    return val

#-------------------------------------------------------------------------------

class ArraySumMethod(object):
    method = 'sum', (types.ArrayKind,), array_dtype_return

    def generic_implement(self, context, args, argtys, retty):
        return imp_numpy_sum(context, args, argtys, retty)

class ArraySumFunction(object):
    function = numpy.sum, (types.ArrayKind,), array_dtype_return

    def generic_implement(self, context, args, argtys, retty):
        return imp_numpy_sum(context, args, argtys, retty)

class ArrayProdMethod(object):
    method = 'prod', (types.ArrayKind,), array_dtype_return

    def generic_implement(self, context, args, argtys, retty):
        return imp_numpy_prod(context, args, argtys, retty)

class ArrayProdFunction(object):
    function = numpy.prod, (types.ArrayKind,), array_dtype_return

    def generic_implement(self, context, args, argtys, retty):
        return imp_numpy_prod(context, args, argtys, retty)

class ArrayShapeAttr(object):
    attribute = 'shape', (types.ArrayKind,), intp_tuple_return

    def generic_implement(self, context, args, argtys, retty):
        ary, = args
        shape = aryutils.getshape(context.builder, ary)
        return retty.desc.llvm_pack(context.builder, shape)

class ArrayStridesAttr(object):
    attribute = 'strides', (types.ArrayKind,), intp_tuple_return

    def generic_implement(self, context, args, argtys, retty):
        ary, = args
        strides = aryutils.getstrides(context.builder, ary)
        return retty.desc.llvm_pack(context.builder, strides)

class ArraySizeAttr(object):
    attribute = 'size', (types.ArrayKind,), types.intp

    def generic_implement(self, context, args, argtys, retty):
        ary, = args
        sz = retty.llvm_const(1)
        for axsz in aryutils.getshape(context.builder, ary):
            sz = context.builder.mul(axsz, sz)
        return sz

class ArrayNdimAttr(object):
    attribute = 'ndim', (types.ArrayKind,), types.intp
    
    def generic_implement(self, context, args, argtys, retty):
        ary, = args
        return retty.llvm_const(aryutils.getndim(ary))

class ArayGetItemIntp(object):
    function = (operator.getitem,
                (types.ArrayKind, types.intp),
                array_dtype_return)

    def generic_implement(self, context, args, argtys, retty):
        ary, idx = args
        aryty = argtys[0]
        indices = wraparound_and_boundcheck(context, ary, [idx])
        return aryutils.getitem(context.builder, ary, indices=indices,
                                order=aryty.desc.order)

class ArrayGetItemTuple(object):
    function = (operator.getitem,
                (types.ArrayKind, getitem_tuple),
                array_complex_slicing_return)
    
    def generic_implement(self, context, args, argtys, retty):
        ary, idx = args
        aryty, idxty = argtys

        builder = context.builder
        markcollapsed = [isinstance(t.desc, types.Integer)
                         for t in idxty.desc.elements]

        nah = lambda x: not bool(x)
        assert len(filter(nah, markcollapsed)) == retty.desc.ndim, \
                    (len(filter(bool, markcollapsed)), retty.desc.ndim)

        indices = idxty.desc.llvm_unpack(builder, idx)

        offsets = []
        lengths = []
        dataptr = aryutils.getdata(builder, ary)
        shapeary = aryutils.getshape(builder, ary)
        stridesary = aryutils.getstrides(builder, ary)
        # for each axis
        for marked, index, shape in zip(markcollapsed, indices, shapeary):
            if marked:  # collapsed
                offsets.append(index)
            else:
                begin, end = types.slice2.llvm_unpack(builder, index)
                begin = axis_wraparound_and_clip(context, begin, shape)
                end = axis_wraparound_and_clip(context, end, shape, plus1=True)
                offsets.append(begin)
                lengths.append(builder.sub(end, begin))

        # build the view
        newdata = aryutils.getpointer(builder, dataptr, shapeary, stridesary,
                                      aryty.desc.order, offsets)
        newstrides = [s for m, s in zip(markcollapsed, stridesary) if not m]
        return aryutils.ndarray(builder, dtype=aryty.desc.element,
                                ndim=retty.desc.ndim, order='A',
                                data=newdata, shape=lengths, strides=newstrides)

class ArrayGetItemFixedArrayIntp(object):
    function = (operator.getitem,
                (types.ArrayKind, getitem_fixedarray_intp),
                array_dtype_return)
    
    def generic_implement(self, context, args, argtys, retty):
        ary, idx = args
        aryty, idxty = argtys
        indices = []

        indexty = types.intp
        ety = idxty.desc.element
        for i in range(idxty.desc.length):
            elem = idxty.desc.llvm_getitem(context.builder, idx, i)
            elem = context.cast(elem, ety, indexty)
            indices.append(elem)

        indices = wraparound_and_boundcheck(context, ary, indices)
        return aryutils.getitem(context.builder, ary, indices=indices,
                                order=aryty.desc.order)

class ArrayGetItemFixedArraySlice2(object):
    function = (operator.getitem,
                (types.ArrayKind, getitem_fixedarray_slice2),
                array_slice_return)
    
    def generic_implement(self, context, args, argtys, retty):
        ary, ls = args
        aryty, lsty = argtys

        begins = []
        ends = []

        builder = context.builder
        slicety = lsty.desc.element
        slices = lsty.desc.llvm_unpack(builder, ls)
        begins, ends = zip(*[slicety.desc.llvm_unpack(builder, sl)
                             for sl in slices])
        
        begins = wraparound_and_clip(context, ary, begins)
        ends = wraparound_and_clip(context, ary, ends, plus1=True)
        res = aryutils.view(builder, ary, begins=begins, ends=ends,
                            order=aryty.desc.order)
        return res

class ArrayGetItemSlice2(object):
    function = (operator.getitem,
                (types.ArrayKind, types.slice2),
                array_slice_return)

    def generic_implement(self, context, args, argtys, retty):
        builder = context.builder
        ary, idx = args
        aryty, idxty = argtys

        begin, end = idxty.desc.llvm_unpack(builder, idx)
        begins = wraparound_and_clip(context, ary, [begin])
        ends = wraparound_and_clip(context, ary, [end], plus1=True)
        res = aryutils.view(builder, ary, begins=begins,
                            ends=ends, order=aryty.desc.order)
        return res

class ArraySetItemIntp(object):
    function = (operator.setitem,
                (types.ArrayKind, types.intp, array_setitem_value),
                types.void)

    def generic_implement(self, context, args, argtys, retty):
        ary, idx, val = args
        aryty, indty, valty = argtys

        val = context.cast(val, valty, aryty.desc.element)
        
        indices = wraparound_and_boundcheck(context, ary, [idx])
        aryutils.setitem(context.builder, ary, indices=indices,
                         order=aryty.desc.order,
                         value=val)

class ArraySetItemTuple(object):
    function = (operator.setitem,
                (types.ArrayKind, types.TupleKind, array_setitem_value),
                types.void)

    def generic_implement(self, context, args, argtys, retty):
        ary, idx, val = args
        aryty, indty, valty = argtys

        indexty = types.intp
        indices = []
        for i, ety in enumerate(indty.desc.elements):
            elem = indty.desc.llvm_getitem(context.builder, idx, i)
            elem = context.cast(elem, ety, indexty)
            indices.append(elem)

        val = context.cast(val, valty, aryty.desc.element)

        indices = wraparound_and_boundcheck(context, ary, indices)
        aryutils.setitem(context.builder, ary, indices=indices,
                         order=aryty.desc.order,
                         value=val)

class ArraySetItemFixedArray(object):
    function = (operator.setitem,
                (types.ArrayKind, types.FixedArrayKind, array_setitem_value),
                types.void)

    def generic_implement(self, context, args, argtys, retty):
        ary, idx, val = args
        aryty, indty, valty = argtys

        indexty = types.intp
        ety = indty.desc.element
        indices = []
        for i in range(indty.desc.length):
            elem = indty.desc.llvm_getitem(context.builder, idx, i)
            elem = context.cast(elem, ety, indexty)
            indices.append(elem)

        val = context.cast(val, valty, aryty.desc.element)

        indices = wraparound_and_boundcheck(context, ary, indices)
        aryutils.setitem(context.builder, ary, indices=indices,
                         order=aryty.desc.order,
                         value=val)
        

extensions = [
ArraySumMethod, ArraySumFunction,
ArrayProdMethod, ArrayProdFunction,
ArrayShapeAttr,
ArrayStridesAttr,
ArraySizeAttr,
ArrayNdimAttr,
ArayGetItemIntp,
ArrayGetItemTuple,
ArrayGetItemFixedArrayIntp,
ArrayGetItemFixedArraySlice2,
ArrayGetItemSlice2,
ArraySetItemIntp,
ArraySetItemTuple,
ArraySetItemFixedArray,
]

