"""
Definition of implementations for the `operator` module.
"""

import operator

from numba.targets.imputils import Registry
from numba.targets import builtins
from numba import types, utils, typing

registry = Registry()
lower = registry.lower


# Redirect the implementation of operator module functions to the
# the corresponding built-in operators.

def map_operator(name, inplace_name, op):
    op_func = getattr(operator, name)

    reverse_args = (op == 'in')
    nargs = (name not in ('pos', 'neg', 'invert', 'not_')) + 1   # must not mix-up binary/unary operators

    @lower(op, *[types.Any]*nargs)
    def binop_impl(context, builder, sig, args):
        if reverse_args:
            args = args[::-1]
            sig = typing.signature(sig.return_type, *sig.args[::-1])

        impl = context.get_function(op_func, sig)
        return impl(builder, args)

        # need a wrapper as inspect.signature(operator.*) does not work, hence cannot
        # .compile_internal op_func directly
        if nargs == 1:
            def wrapper(x):
                return op_func(x)
        else:
            def wrapper(x, y):
                return op_func(x, y)

        return context.compile_internal(builder, wrapper, sig, args)

    if inplace_name:
        inplace_op_func = getattr(operator, inplace_name)

        @lower(op + '=', *[types.Any]*nargs)
        def binop_inplace_impl(context, builder, sig, args):
            first = sig.args[0]
            op_func_ = inplace_op_func #if first.mutable else op_func

            impl = context.get_function(op_func_, sig)
            return impl(builder, args)

            if nargs == 1:
                def wrapper(x):
                    return op_func_(x)
            else:
                def wrapper(x, y):
                    return op_func_(x, y)

            return context.compile_internal(builder, wrapper, sig, args)


for name, inplace_name, op in utils.operator_map:
    map_operator(name, inplace_name, op)
