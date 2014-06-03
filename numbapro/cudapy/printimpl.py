from __future__ import print_function, absolute_import, division
from llvm.core import Type, Constant
from numba import types, typing, cgutils
from numba.targets.imputils import implement, Registry
from . import nvvmutils

registry = Registry()
register = registry.register

voidptr = Type.pointer(Type.int(8))


def int_print_impl(context, builder, sig, args):
    [x] = args
    [srctype] = sig.args
    mod = cgutils.get_module(builder)
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


for ty in types.integer_domain:
    register(implement(types.print_item_type, ty)(int_print_impl))


def real_print_impl(context, builder, sig, args):
    [x] = args
    [srctype] = sig.args
    mod = cgutils.get_module(builder)
    vprint = nvvmutils.declare_vprint(mod)
    rawfmt = "%f"
    dsttype = types.float64
    fmt = context.insert_string_const_addrspace(builder, rawfmt)
    lld = context.cast(builder, x, srctype, dsttype)
    valptr = cgutils.alloca_once(builder, context.get_value_type(dsttype))
    builder.store(lld, valptr)
    builder.call(vprint, [fmt, builder.bitcast(valptr, voidptr)])
    return context.get_dummy_value()


for ty in types.real_domain:
    register(implement(types.print_item_type, ty)(real_print_impl))


@register
@implement(types.print_type, types.VarArg)
def print_varargs(context, builder, sig, args):
    mod = cgutils.get_module(builder)
    vprint = nvvmutils.declare_vprint(mod)
    for i, (argtype, argval) in enumerate(zip(sig.args, args)):
        signature = typing.signature(types.none, argtype)
        imp = context.get_function(types.print_item_type, signature)
        imp(builder, [argval])
        if i == len(args) - 1:
            eos = context.insert_string_const_addrspace(builder, " ")
        else:
            eos = context.insert_string_const_addrspace(builder, "\n")

        builder.call(vprint, (eos, Constant.null(voidptr)))

    return context.get_dummy_value()
