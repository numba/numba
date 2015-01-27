"""
This file implements print functionality for the CPU.
"""
from __future__ import print_function, absolute_import, division
from llvmlite.llvmpy.core import Type
from numba import types, typing, cgutils
from numba.targets.imputils import implement, Registry

registry = Registry()
register = registry.register


# FIXME: the current implementation relies on CPython API even in
#        nopython mode.


def int_print_impl(context, builder, sig, args):
    [x] = args
    py = context.get_python_api(builder)
    szval = context.cast(builder, x, sig.args[0], types.intp)
    intobj = py.long_from_ssize_t(szval)
    py.print_object(intobj)
    py.decref(intobj)
    return context.get_dummy_value()


for ty in types.integer_domain:
    register(implement(types.print_item_type, ty)(int_print_impl))


def real_print_impl(context, builder, sig, args):
    [x] = args
    py = context.get_python_api(builder)
    szval = context.cast(builder, x, sig.args[0], types.float64)
    intobj = py.float_from_double(szval)
    py.print_object(intobj)
    py.decref(intobj)
    return context.get_dummy_value()


for ty in types.real_domain:
    register(implement(types.print_item_type, ty)(real_print_impl))


@register
@implement(types.print_item_type, types.Kind(types.CharSeq))
def print_charseq(context, builder, sig, args):
    [x] = args
    py = context.get_python_api(builder)
    xp = cgutils.alloca_once(builder, x.type)
    builder.store(x, xp)
    byteptr = builder.bitcast(xp, Type.pointer(Type.int(8)))
    size = context.get_constant(types.intp, x.type.elements[0].count)
    cstr = py.bytes_from_string_and_size(byteptr, size)
    py.print_object(cstr)
    py.decref(cstr)
    return context.get_dummy_value()


@register
@implement(types.print_type, types.VarArg(types.Any))
def print_varargs(context, builder, sig, args):
    py = context.get_python_api(builder)
    for i, (argtype, argval) in enumerate(zip(sig.args, args)):
        signature = typing.signature(types.none, argtype)
        imp = context.get_function(types.print_item_type, signature)
        imp(builder, [argval])
        if i == len(args) - 1:
            py.print_string('\n')
        else:
            py.print_string(' ')

    return context.get_dummy_value()
