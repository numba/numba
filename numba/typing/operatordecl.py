"""
Typing declarations for the operator module.
"""

import operator

from numba import types
from numba import utils
from numba.typing.templates import (ConcreteTemplate, AbstractTemplate,
                                    signature, Registry)

registry = Registry()
infer_getattr = registry.register_attr
infer_global = registry.register_global


class MappedOperator(AbstractTemplate):

    # Whether the args to the operator and the operator module func are reversed
    reverse_args = False

    def generic(self, args, kws):
        assert not kws
        args = args[::-1] if self.reverse_args else args
        sig = self.context.resolve_function_type(self.op, args, kws)
        if self.reverse_args and sig is not None:
            sig = signature(sig.return_type, *sig.args[::-1])
        return sig


class MappedInplaceOperator(AbstractTemplate):

    def generic(self, args, kws):
        assert not kws
        if not args:
            return
        first = args[0]
        op = self.mutable_op if first.mutable else self.immutable_op
        return self.context.resolve_function_type(op, args, kws)


# Redirect all functions in the operator module to the corresponding
# built-in operators.

for name, inplace_name, op in utils.operator_map:
    op_func = getattr(operator, name)
    op_type = type('Operator_' + name, (MappedOperator,),
                   {'key': op_func, 'op': op,
                    'reverse_args': op == 'in'})
    infer_global(op_func, types.Function(op_type))

    if inplace_name:
        op_func = getattr(operator, inplace_name)
        op_type = type('Operator_' + inplace_name, (MappedInplaceOperator,),
                       {'key': op_func,
                        'mutable_op': op + '=',
                        'immutable_op': op})
        infer_global(op_func, types.Function(op_type))
