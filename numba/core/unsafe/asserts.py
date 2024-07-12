"""
Compile-time assertions.
"""
from inspect import signature
from numba.core import types, errors
from numba.core.extending import intrinsic, sentry_literal_args


_sig_of_assert = signature(lambda msg: None)


class BailTypingError(AssertionError):
    pass


@intrinsic(prefer_literal=True)
def assert_not_typing(typingctx, msg):
    """assert_not_typing(msg: str)

    Raises AssertionError during typing
    """
    if isinstance(msg, types.Literal):
        raise BailTypingError(f"typing should fail: {msg.literal_value!r}")
    else:
        raise errors.TypingError(f"{msg} must be literal")


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
