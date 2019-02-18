"""
This file provides internal compiler utilities that support certain special
operations with bytes and workarounds for limitations enforced in userland.
"""

from numba.extending import intrinsic
from llvmlite import ir
from numba import types


@intrinsic
def grab_byte(typingctx, data, offset):
    # returns a byte at a given offset in data
    def impl(context, builder, signature, args):
        data, idx = args
        ptr = builder.bitcast(data, ir.IntType(8).as_pointer())
        ch = builder.load(builder.gep(ptr, [idx]))
        return ch

    sig = types.uint8(types.voidptr, types.intp)
    return sig, impl


@intrinsic
def grab_uint64_t(typingctx, data, offset):
    # returns a uint64_t at a given offset in data
    def impl(context, builder, signature, args):
        data, idx = args
        ptr = builder.bitcast(data, ir.IntType(64).as_pointer())
        ch = builder.load(builder.gep(ptr, [idx]))
        return ch
    sig = types.uint64(types.voidptr, types.intp)
    return sig, impl
