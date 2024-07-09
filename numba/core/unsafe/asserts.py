"""
Compile-time assertions.
"""
from inspect import signature
from numba.core import types
from numba.core.extending import intrinsic, sentry_literal_args


_sig_of_assert = signature(lambda msg: None)


@intrinsic(prefer_literal=True)
def assert_not_typing(typingctx, msg):
    """assert_not_typing(msg: str)

    Raises AssertionError during typing
    """
    sentry_literal_args(_sig_of_assert, ("msg",), (msg,), {})
    raise AssertionError(f"typing should fail: {msg.literal_value!r}")


@intrinsic
def assert_not_lowering(typingctx, msg):
    """assert_not_lowering(msg: str)

    Raises AssertionError during lowering
    """
    def codegen(context, builder, signature, args):
        raise AssertionError(literal_msg)

    sentry_literal_args(_sig_of_assert, ("msg",), (msg,), {})
    literal_msg = f"lowering should fail: {msg.literal_value}"
    return types.none(msg), codegen
