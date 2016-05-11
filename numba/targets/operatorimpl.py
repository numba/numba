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

    @lower(op_func, types.VarArg(types.Any))
    def binop_impl(context, builder, sig, args):
        if reverse_args:
            args = args[::-1]
            sig = typing.signature(sig.return_type, *sig.args[::-1])
        impl = context.get_function(op, sig)
        return impl(builder, args)

    if inplace_name:
        op_func = getattr(operator, inplace_name)

        @lower(op_func, types.VarArg(types.Any))
        def binop_inplace_impl(context, builder, sig, args):
            first = sig.args[0]
            if first.mutable:
                impl = context.get_function(op + '=', sig)
            else:
                impl = context.get_function(op, sig)
            return impl(builder, args)


for name, inplace_name, op in utils.operator_map:
    map_operator(name, inplace_name, op)
