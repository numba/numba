"""
Implementation of various iterable and iterator types.
"""

from llvm.core import Type, Constant

from numba import errcode
from numba import types, typing, cgutils
from numba.targets.imputils import (
    builtin, implement, iternext_impl, call_iternext, struct_factory)


@builtin
@implement('getiter', types.Kind(types.IteratorType))
def iterator_getiter(context, builder, sig, args):
    [it] = args
    return it


#-------------------------------------------------------------------------------
# builtin `enumerate` implementation

@struct_factory(types.EnumerateType)
def make_enumerate_cls(enum_type):
    """
    Return the Structure representation of the given *enum_type* (an
    instance of types.EnumerateType).
    """

    class Enumerate(cgutils.Structure):
        _fields = [('count', types.CPointer(types.intp)),
                   ('iter', enum_type.source_type)]

    return Enumerate

@builtin
@implement(enumerate, types.Kind(types.IterableType))
def make_enumerate_object(context, builder, sig, args):
    [srcty] = sig.args
    [src] = args

    getiter_sig = typing.signature(srcty.iterator_type, srcty)
    getiter_impl = context.get_function('getiter', getiter_sig)
    iterobj = getiter_impl(builder, (src,))

    enumcls = make_enumerate_cls(sig.return_type)
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

    enumcls = make_enumerate_cls(enumty)
    enum = enumcls(context, builder, value=enum)

    count = builder.load(enum.count)
    ncount = builder.add(count, context.get_constant(types.intp, 1))
    builder.store(ncount, enum.count)

    srcres = call_iternext(context, builder, enumty.source_type, enum.iter)
    is_valid = srcres.is_valid()
    result.set_valid(is_valid)

    with cgutils.ifthen(builder, is_valid):
        srcval = srcres.yielded_value()
        result.yield_(cgutils.make_anonymous_struct(builder, [count, srcval]))
