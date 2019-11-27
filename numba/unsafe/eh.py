"""
Exception handling intrinsics.
"""

from numba import types, cgutils, errors
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


@intrinsic
def mark_try_block(typingctx):
    def codegen(context, builder, signature, args):
        nrt = context.nrt
        nrt.eh_try(builder)
        return context.get_dummy_value()

    restype = types.none
    return restype(), codegen


@intrinsic
def end_try_block(typingctx):
    def codegen(context, builder, signature, args):
        nrt = context.nrt
        nrt.eh_end_try(builder)
        return context.get_dummy_value()

    restype = types.none
    return restype(), codegen


@intrinsic
def exception_match(typingctx, exc_value, exc_class):
    if exc_class.exc_class is not Exception:
        msg = "Exception matching is limited to {}"
        raise errors.UnsupportedError(msg.format(Exception))

    def codegen(context, builder, signature, args):
        return cgutils.true_bit

    restype = types.boolean
    return restype(exc_value, exc_class), codegen
