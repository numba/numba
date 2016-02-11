from __future__ import print_function, absolute_import, division

from llvmlite.llvmpy.core import Type, Constant

from numba import types, typing, cgutils
from numba.targets.imputils import Registry
from . import nvvmutils

registry = Registry()
lower = registry.lower

voidptr = Type.pointer(Type.int(8))

@lower("print_item", types.Integer)
def int_print_impl(context, builder, sig, args):
    [x] = args
    [srctype] = sig.args
    mod = builder.module
    vprint = nvvmutils.declare_vprint(mod)
    if srctype in types.unsigned_domain:
        rawfmt = "%llu"
        dsttype = types.uint64
    else:
        rawfmt = "%lld"
        dsttype = types.int64
    fmt = context.insert_string_const_addrspace(builder, rawfmt)
    lld = context.cast(builder, x, srctype, dsttype)
    valptr = cgutils.alloca_once(builder, context.get_value_type(dsttype))
    builder.store(lld, valptr)
    builder.call(vprint, [fmt, builder.bitcast(valptr, voidptr)])
    return context.get_dummy_value()


@lower("print_item", types.Float)
def real_print_impl(context, builder, sig, args):
    [x] = args
    [srctype] = sig.args
    mod = builder.module
    vprint = nvvmutils.declare_vprint(mod)
    rawfmt = "%f"
    dsttype = types.float64
    fmt = context.insert_string_const_addrspace(builder, rawfmt)
    lld = context.cast(builder, x, srctype, dsttype)
    valptr = cgutils.alloca_once(builder, context.get_value_type(dsttype))
    builder.store(lld, valptr)
    builder.call(vprint, [fmt, builder.bitcast(valptr, voidptr)])
    return context.get_dummy_value()


@lower(print, types.VarArg(types.Any))
def print_varargs(context, builder, sig, args):
    """This function is a generic 'print' wrapper for arbitrary types.
    It dispatches to the appropriate 'print' implementations above
    depending on the detected real types in the signature."""

    mod = builder.module
    vprint = nvvmutils.declare_vprint(mod)
    sep = context.insert_string_const_addrspace(builder, " ")
    eol = context.insert_string_const_addrspace(builder, "\n")

    for i, (argtype, argval) in enumerate(zip(sig.args, args)):
        signature = typing.signature(types.none, argtype)
        imp = context.get_function("print_item", signature)
        imp(builder, [argval])
        if i < len(args) - 1:
            builder.call(vprint, (sep, Constant.null(voidptr)))

    builder.call(vprint, (eol, Constant.null(voidptr)))

    return context.get_dummy_value()
