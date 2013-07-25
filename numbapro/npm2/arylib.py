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

    begins = (0,) * aryty.desc.ndim
    ends = aryutils.getshape(builder, ary)
    steps = (1,) * aryty.desc.ndim

    sum = builder.alloca(elemty.llvm_as_value())
    builder.store(elemty.llvm_const(0), sum)
    with cgutils.loop_nest(builder, begins, ends, steps) as indices:
        val = aryutils.getitem(builder, ary, indices=indices,
                                             order=aryty.desc.order)
        # XXX: move to next level and reuse NPM to compile this
        assert isinstance(elemty.desc, types.Float)
        do_add = context.imp.lookup(operator.add, (elemty, elemty))
        new_sum = do_add(context, (context.builder.load(sum), val))
        builder.store(new_sum, sum)

    return builder.load(sum)

def numpy_sum_return(args):
    ary, = args
    return ary.desc.element

class ArraySumMethod(object):
    method = 'sum', (types.ArrayKind,), numpy_sum_return

    def generic_implement(self, context, args, argtys, retty):
        return imp_numpy_sum(context, args, argtys, retty)

class ArraySumFunction(object):
    function = numpy.sum, (types.ArrayKind,), numpy_sum_return

    def generic_implement(self, context, args, argtys, retty):
        return imp_numpy_sum(context, args, argtys, retty)

extensions = [ArraySumMethod, ArraySumFunction]

