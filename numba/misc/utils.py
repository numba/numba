from numba.core import cgutils, config, types
from numba.core.withcontexts import bypass_context
from numba.core.extending import intrinsic, register_jitable
import sys


# printf
@intrinsic
def _printf(typingctx, format_type, *args):
    """printf that can be called from Numba jit-decorated functions.
    """
    if isinstance(format_type, types.StringLiteral):
        sig = types.void(format_type, types.BaseTuple.from_types(args))

        def codegen(context, builder, signature, args):
            cgutils.printf(builder, format_type.literal_value, *args[1:])
        return sig, codegen


@register_jitable
def printf(format_type, *args):
    if config.DISABLE_JIT:
        with bypass_context:
            print(format_type % args, end='')
    else:
        _printf(format_type, args)


# fflush
@intrinsic
def _fflush(typingctx):
    sig = types.void(types.void)

    def codegen(context, builder, signature, args):
        cgutils.fflush(builder)

    return sig, codegen


@register_jitable
def fflush():
    if config.DISABLE_JIT:
        with bypass_context:
            sys.stdout.flush()
            sys.stderr.flush()
    else:
        _fflush()
