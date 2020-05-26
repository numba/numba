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

        def codegen(context, builder, signature, fn_args):
            mod = builder.module
            value = fn_args[1].value # StructLiteral
            for i, arg in enumerate(*args):
                if isinstance(arg, types.StringLiteral):
                    lv = arg.literal_value
                    const_str = context.insert_const_string(mod, lv)
                    value = builder.insert_value(value, const_str, i)

            # Replace %c formatter by %s, since it doesn't work here
            ft = format_type.literal_value.replace('%c', '%s')
            cgutils.printf(builder, ft, value)
        return sig, codegen


@register_jitable
def printf(format_type, *args):
    if config.DISABLE_JIT:
        if '%p' in format_type:
            raise ValueError(
                '"%p" is not supported when JIT is disabled.'
                '\nReplace the formatter by "%#x" and use the identity (id)'
                ' of the object.\nFor example:\n'
                '\n>>> arg = "hello world"'
                '\n>>> print("%#x" % id(arg))'
                '\n0x10531d1f0\n')
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
