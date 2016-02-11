"""
This file implements print functionality for the CPU.
"""
from __future__ import print_function, absolute_import, division
from llvmlite.llvmpy.core import Type
from numba import types, typing, cgutils
from numba.targets.imputils import Registry, impl_ret_untracked

registry = Registry()
lower = registry.lower


# FIXME: the current implementation relies on CPython API even in
#        nopython mode.


@lower("print_item", types.Integer)
def int_print_impl(context, builder, sig, args):
    [x] = args
    py = context.get_python_api(builder)
    if sig.args[0].signed:
        intobj = py.long_from_signed_int(x)
    else:
        intobj = py.long_from_unsigned_int(x)
    py.print_object(intobj)
    py.decref(intobj)
    res = context.get_dummy_value()
    return impl_ret_untracked(context, builder, sig.return_type, res)


@lower("print_item", types.Float)
def real_print_impl(context, builder, sig, args):
    [x] = args
    py = context.get_python_api(builder)
    szval = context.cast(builder, x, sig.args[0], types.float64)
    intobj = py.float_from_double(szval)
    py.print_object(intobj)
    py.decref(intobj)
    res = context.get_dummy_value()
    return impl_ret_untracked(context, builder, sig.return_type, res)


@lower("print_item", types.Boolean)
def bool_print_impl(context, builder, sig, args):
    [x] = args
    py = context.get_python_api(builder)
    boolobj = py.bool_from_bool(x)
    py.print_object(boolobj)
    py.decref(boolobj)
    res = context.get_dummy_value()
    return impl_ret_untracked(context, builder, sig.return_type, res)


@lower("print_item", types.CharSeq)
def print_charseq(context, builder, sig, args):
    [tx] = sig.args
    [x] = args
    py = context.get_python_api(builder)
    xp = cgutils.alloca_once(builder, x.type)
    builder.store(x, xp)
    byteptr = builder.bitcast(xp, Type.pointer(Type.int(8)))
    size = context.get_constant(types.intp, tx.count)
    cstr = py.bytes_from_string_and_size(byteptr, size)
    py.print_object(cstr)
    py.decref(cstr)
    res = context.get_dummy_value()
    return impl_ret_untracked(context, builder, sig.return_type, res)


@lower(print, types.VarArg(types.Any))
def print_varargs(context, builder, sig, args):
    py = context.get_python_api(builder)
    for i, (argtype, argval) in enumerate(zip(sig.args, args)):
        signature = typing.signature(types.none, argtype)
        imp = context.get_function("print_item", signature)
        imp(builder, [argval])
        if i < len(args) - 1:
            py.print_string(' ')
    py.print_string('\n')

    res = context.get_dummy_value()
    return impl_ret_untracked(context, builder, sig.return_type, res)
