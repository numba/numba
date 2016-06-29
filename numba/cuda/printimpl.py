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
    if srctype in types.unsigned_domain:
        rawfmt = "%llu"
        dsttype = types.uint64
    else:
        rawfmt = "%lld"
        dsttype = types.int64
    fmt = context.insert_string_const_addrspace(builder, rawfmt)
    lld = context.cast(builder, x, srctype, dsttype)
    return rawfmt, [lld]


@lower("print_item", types.Float)
def real_print_impl(context, builder, sig, args):
    [x] = args
    [srctype] = sig.args
    rawfmt = "%f"
    dsttype = types.float64
    lld = context.cast(builder, x, srctype, dsttype)
    return rawfmt, [lld]


@lower("print_item", types.Const)
def const_print_impl(context, builder, sig, args):
    ty, = sig.args
    pyval = ty.value
    assert isinstance(pyval, str)  # Ensured by lowering
    rawfmt = "%s"
    val = context.insert_string_const_addrspace(builder, pyval)
    return rawfmt, [val]


@lower(print, types.VarArg(types.Any))
def print_varargs(context, builder, sig, args):
    """This function is a generic 'print' wrapper for arbitrary types.
    It dispatches to the appropriate 'print' implementations above
    depending on the detected real types in the signature."""

    vprint = nvvmutils.declare_vprint(builder.module)

    formats = []
    values = []

    for i, (argtype, argval) in enumerate(zip(sig.args, args)):
        # XXX The return type is a lie: "print_item" doesn't return
        # a LLVM value at all.
        signature = typing.signature(types.none, argtype)
        impl = context.get_function("print_item", signature)
        argfmt, argvals = impl(builder, (argval,))
        formats.append(argfmt)
        values.extend(argvals)

    rawfmt = " ".join(formats) + "\n"
    fmt = context.insert_string_const_addrspace(builder, rawfmt)
    array = cgutils.make_anonymous_struct(builder, values)
    arrayptr = cgutils.alloca_once_value(builder, array)

    vprint = nvvmutils.declare_vprint(builder.module)
    builder.call(vprint, (fmt, builder.bitcast(arrayptr, voidptr)))

    return context.get_dummy_value()
