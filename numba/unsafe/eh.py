"""
Exception handling intrinsics.
"""

from numba import types
from numba.extending import intrinsic


@intrinsic
def exception_check(typingctx):
    """Intrinsic to check if an exception is raised
    """
    def codegen(context, builder, signature, args):
        nrt = context.nrt
        return nrt.eh_check(builder)

    restype = types.boolean
    return restype(), codegen
