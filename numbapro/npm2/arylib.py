'''
Implements numpy array functions.
'''
import operator
import numpy
from . import cgutils, aryutils, types, imlib

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

def numpy_dtype_return(args):
    ary, = args
    return ary.desc.element

def intp_tuple_return(args):
    ary = args[0]
    return types.tupletype(*([types.intp] * ary.desc.ndim))

#-------------------------------------------------------------------------------

class ArraySumMethod(object):
    method = 'sum', (types.ArrayKind,), numpy_dtype_return

    def generic_implement(self, context, args, argtys, retty):
        return imp_numpy_sum(context, args, argtys, retty)

class ArraySumFunction(object):
    function = numpy.sum, (types.ArrayKind,), numpy_dtype_return

    def generic_implement(self, context, args, argtys, retty):
        return imp_numpy_sum(context, args, argtys, retty)

class ArrayProdMethod(object):
    method = 'prod', (types.ArrayKind,), numpy_dtype_return

    def generic_implement(self, context, args, argtys, retty):
        return imp_numpy_prod(context, args, argtys, retty)

class ArrayProdFunction(object):
    function = numpy.prod, (types.ArrayKind,), numpy_dtype_return

    def generic_implement(self, context, args, argtys, retty):
        return imp_numpy_prod(context, args, argtys, retty)

class ArrayShapeAttr(object):
    attribute = 'shape', (types.ArrayKind,), intp_tuple_return

    def generic_implement(self, context, args, argtys, retty):
        ary, = args
        shape = aryutils.getshape(context.builder, ary)
        return retty.desc.llvm_pack(context.builder, shape)

extensions = [
ArraySumMethod, ArraySumFunction,
ArrayProdMethod, ArrayProdFunction,
ArrayShapeAttr,
]

