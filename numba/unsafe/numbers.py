""" This module provides the unsafe things for targets/numbers.py
"""
from .. import types
from ..extending import intrinsic

from llvmlite import ir


@intrinsic
def viewer(tyctx, val, viewty):
    """ Bitcast a scalar 'val' to the given type 'viewty'. """
    bits = val.bitwidth
    if isinstance(viewty.dtype, types.Integer):
        bitcastty = ir.IntType(bits)
    elif isinstance(viewty.dtype, types.Float):
        bitcastty = ir.FloatType() if bits == 32 else ir.DoubleType()
    else:
        assert 0, "unreachable"

    def codegen(cgctx, builder, typ, args):
        flt = args[0]
        return builder.bitcast(flt, bitcastty)
    retty = viewty.dtype
    sig = retty(val, viewty)
    return sig, codegen
