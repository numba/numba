"""
Exception handling intrinsics.
"""

from numba.core import types
from numba.core.extending import intrinsic


@intrinsic
def exception_check(typingctx):
    """An intrinsic to check if an exception is raised
    """
    def codegen(context, builder, signature, args):
        nrt = context.nrt
        return nrt.eh_check(builder)

    restype = types.boolean
    return restype(), codegen


@intrinsic
def mark_try_block(typingctx):
    """An intrinsic to mark the start of a *try* block.
    """
    def codegen(context, builder, signature, args):
        nrt = context.nrt
        nrt.eh_try(builder)
        return context.get_dummy_value()

    restype = types.none
    return restype(), codegen


@intrinsic
def end_try_block(typingctx):
    """An intrinsic to mark the end of a *try* block.
    """
    def codegen(context, builder, signature, args):
        nrt = context.nrt
        nrt.eh_end_try(builder)
        return context.get_dummy_value()

    restype = types.none
    return restype(), codegen

@intrinsic
def end_matching_block(typingctx):
    """An intrinsic to mark the end of
    exception matching blocks.
    """
    def codegen(context, builder, signature, args):
        context.call_conv.reraise_try(builder)
        return context.get_dummy_value()

    restype = types.none
    return restype(), codegen


@intrinsic
def exception_match(typingctx, exc_value, exc_class):
    """Basically do ``isinstance(exc_value, exc_class)`` for exception objects.
    Used in ``except Exception:`` syntax.
    """
    def codegen(context, builder, signature, args):
        _, exc_class = signature.args
        res = context.call_conv.match_user_exc(builder, exc_class.exc_class)
        return res

    restype = types.boolean
    return restype(exc_value, exc_class), codegen
