"""
Implementation of various iterable and iterator types.
"""

from llvm.core import Type, Constant

from numba import errcode
from numba import types, typing, cgutils
from numba.targets.imputils import builtin, implement, iternext_impl, call_iternext


@builtin
@implement('getiter', types.Kind(types.IteratorType))
def iterator_getiter(context, builder, sig, args):
    [it] = args
    return it


#-------------------------------------------------------------------------------
# builtin `enumerate` implementation

def make_enumerate_cls(source_iterator):

    class Enumerate(cgutils.Structure):
        _fields = [('count', types.CPointer(types.intp)),
                   ('iter', source_iterator)]

    return Enumerate

@builtin
@implement(enumerate, types.Kind(types.IterableType))
def make_enumerate_object(context, builder, sig, args):
    [srcty] = sig.args
    [src] = args

    getiter_sig = typing.signature(srcty.iterator_type, srcty)
    getiter_impl = context.get_function('getiter', getiter_sig)
    iterobj = getiter_impl(builder, (src,))

    enumcls = make_enumerate_cls(srcty.iterator_type)
    enum = enumcls(context, builder)

    zero = context.get_constant(types.intp, 0)
    countptr = cgutils.alloca_once(builder, zero.type)
    builder.store(zero, countptr)

    enum.count = countptr
    enum.iter = iterobj

    return enum._getvalue()

@builtin
@implement('iternext', types.Kind(types.EnumerateType))
@iternext_impl
def iternext_enumerate(context, builder, sig, args, result):
    [enumty] = sig.args
    [enum] = args

    enumcls = make_enumerate_cls(enumty.source_type)
    enum = enumcls(context, builder, value=enum)

    count = builder.load(enum.count)
    ncount = builder.add(count, context.get_constant(types.intp, 1))
    builder.store(ncount, enum.count)

    srcres = call_iternext(context, builder, enumty.source_type, enum.iter)
    is_valid = srcres.is_valid()
    result.set_valid(is_valid)

    with cgutils.ifthen(builder, is_valid):
        srcval = srcres.yielded_value()
        struct_type = Type.struct([count.type, srcval.type])
        struct_val = Constant.undef(struct_type)
        struct_val = builder.insert_value(struct_val, count, 0)
        struct_val = builder.insert_value(struct_val, srcval, 1)
        result.yield_(struct_val)
