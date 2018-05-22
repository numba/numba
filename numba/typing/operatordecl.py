"""
Typing declarations for the operator module.
"""

import operator

from numba import types
from numba import utils, jit
from numba.typing.templates import (CallableTemplate, AbstractTemplate,
                                    signature, Registry)

registry = Registry()
infer_getattr = registry.register_attr
infer = registry.register


class MappedOperator(AbstractTemplate):

    # Whether the args to the operator and the operator module func are reversed
    reverse_args = False

    def generic(self, args, kws):
        assert not kws
        if len(args) != self.nargs:
            return
        args = args[::-1] if self.reverse_args else args
        sig = self.context.resolve_function_type(self.op, args, kws)
        if self.reverse_args and sig is not None:
            sig = signature(sig.return_type, *sig.args[::-1])
        return sig


# class MappedInplaceOperator(AbstractTemplate):
#
#     def generic(self, args, kws):
#         assert not kws
#         if len(args) != self.nargs:
#             return
#         first = args[0]
#         op = self.mutable_op if first.mutable else self.immutable_op
#         return self.context.resolve_function_type(op, args, kws)


# Redirect all built-in operators to the corresponding functions in the operator module.
#

for name, inplace_name, op in utils.operator_map:
    op_func = getattr(operator, name)
    nargs = (name not in ('pos', 'neg', 'invert', 'not_')) + 1

    op_type = type('Operator_' + name, (MappedOperator,),
                   {'key': op, 'op': op_func,
                    'reverse_args': op == 'in',
                    'nargs': nargs})
    infer(op_type)

    if inplace_name:
        op_func = getattr(operator, inplace_name)
        nargs = (name not in ('pos', 'neg', 'invert', 'not_')) + 1

        op_type = type('Operator_' + inplace_name, (MappedOperator,),
                       {'key': op + '=', 'op': op_func,
                        'reverse_args': op == 'in',
                        'nargs': nargs})
        infer(op_type)
