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

def array_dtype_return(args):
    return args[0].desc.element

def intp_tuple_return(args):
    ary = args[0]
    return types.tupletype(*([types.intp] * ary.desc.ndim))

def array_setitem_value(args):
    ary = args[0]
    return ary.desc.element

def array_slice_return(args):
    ary, idx = args
    assert isinstance(idx.desc, types.Slice)
    if ary.desc.ndim != 1:
        raise TypeError('expecting 1d array')
    start = idx.desc.has_start
    stop = idx.desc.has_stop
    step = idx.desc.has_step
    order = ary.desc.order
    if (not start and not stop and not step): # [:]
        return ary.desc.copy(order='%sS' % order[0])
    elif order != 'A' and (not start and stop and not step): # [:x]
        return ary.desc.copy(order='%sS' % order[0])
    elif order != 'A' and (start and not stop and not step): # [x:]
        return ary.desc.copy(order='%sS' % order[0])
    elif order != 'A' and (start and stop and not step): # [x:y]
        return ary.desc.copy(order='%sS' % order[0])
    raise TypeError('unsupported slicing operation')

def boundcheck(context, ary, indices):
    assert isinstance(indices, (tuple, list))
    if not context.flags.no_boundcheck:
        aryutils.boundcheck(context.builder, context.raises, ary, indices)

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
        boundcheck(context, ary, [idx])
        return aryutils.getitem(context.builder, ary, indices=[idx],
                                order=aryty.desc.order)

class ArrayGetItemTuple(object):
    function = (operator.getitem,
                (types.ArrayKind, types.TupleKind),
                array_dtype_return)
    
    def generic_implement(self, context, args, argtys, retty):
        ary, idx = args
        aryty, idxty = argtys
        indices = []

        indexty = types.intp
        for i, ety in enumerate(idxty.desc.elements):
            elem = idxty.desc.llvm_getitem(context.builder, idx, i)
            elem = context.cast(elem, ety, indexty)
            indices.append(elem)

        boundcheck(context, ary, indices)
        return aryutils.getitem(context.builder, ary, indices=indices,
                                order=aryty.desc.order)

class ArrayGetItemFixedArray(object):
    function = (operator.getitem,
                (types.ArrayKind, types.FixedArrayKind),
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

        boundcheck(context, ary, indices)
        return aryutils.getitem(context.builder, ary, indices=indices,
                                order=aryty.desc.order)

class ArrayGetItemSlice(object):
    function = (operator.getitem,
                (types.ArrayKind, types.SliceKind),
                array_slice_return)

    def generic_implement(self, context, args, argtys, retty):
        builder = context.builder
        ary, idx = args
        aryty, idxty = argtys

        has_start = idxty.desc.has_start
        has_stop = idxty.desc.has_stop
        has_step = idxty.desc.has_step
        if not has_start and not has_stop and not has_step: # [:]
            return ary
        elif not has_start and has_stop and not has_step:   # [:x]
            end = builder.extract_value(idx, 1)
            res = aryutils.view(builder, ary, begins=[0], ends=[end],
                                order=aryty.desc.order)
            return res
        elif has_start and not has_stop and not has_step:   # [x:]
            begin = builder.extract_value(idx, 0)
            ends = aryutils.getshape(builder, ary)
            res = aryutils.view(builder, ary, begins=[begin],
                                ends=ends, order=aryty.desc.order)
            return res
        elif has_start and has_stop and not has_step:       # [x:y]
            begin = builder.extract_value(idx, 0)
            end = builder.extract_value(idx, 1)
            res = aryutils.view(builder, ary, begins=[begin],
                                ends=[end], order=aryty.desc.order)
            return res
        else:
            raise NotImplementedError

class ArraySetItemIntp(object):
    function = (operator.setitem,
                (types.ArrayKind, types.intp, array_setitem_value),
                types.void)

    def generic_implement(self, context, args, argtys, retty):
        ary, idx, val = args
        aryty, indty, valty = argtys

        val = context.cast(val, valty, aryty.desc.element)
        
        boundcheck(context, ary, [idx])
        aryutils.setitem(context.builder, ary, indices=[idx],
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

        boundcheck(context, ary, indices)
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

        boundcheck(context, ary, indices)
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
ArrayGetItemFixedArray,
ArrayGetItemSlice,
ArraySetItemIntp,
ArraySetItemTuple,
ArraySetItemFixedArray,
]

