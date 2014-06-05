import operator
from numba import types
from numba import utils
from numba.typing.templates import (AttributeTemplate, ConcreteTemplate,
                                    signature, Registry)

registry = Registry()
builtin_attr = registry.register_attr
builtin_global = registry.register_global

class OperatorModuleAttribute(AttributeTemplate):
    key = types.Module(operator)

class Operator_unary(ConcreteTemplate):
    cases = [
        signature(types.int64, types.int64),
        signature(types.uint64, types.uint64),
        signature(types.float32, types.float32),
        signature(types.float64, types.float64),
    ]

class Operator_binary(ConcreteTemplate):
    cases = [
        signature(types.int64, types.int64, types.int64),
        signature(types.uint64, types.uint64, types.uint64),
        signature(types.float32, types.float32, types.float32),
        signature(types.float64, types.float64, types.float64),
    ]

def create_resolve_method(op):
    def resolve_op(self, mod):
        return types.Function(op)
    return resolve_op


unary_operators = ['neg', 'invert']

for op in unary_operators:
    op_type = type('Operator_' + op, (Operator_unary,), {'key':getattr(operator, op)})
    setattr(OperatorModuleAttribute, 'resolve_' + op,  create_resolve_method(op_type))
    builtin_global(getattr(operator, op), types.Function(op_type))


binary_operators = ['add', 'sub', 'mul', 'div', 'floordiv', 'truediv', 'mod', 'pow',
    'eq', 'ne', 'lt', 'le', 'gt', 'ge', 'and_', 'or_', 'xor',
    'lshift', 'rshift']

if utils.IS_PY3:
    binary_operators.remove('div')
    
for op in binary_operators:
    op_type = type('Operator_' + op, (Operator_binary,), {'key':getattr(operator, op)})
    setattr(OperatorModuleAttribute, 'resolve_' + op,  create_resolve_method(op_type))
    builtin_global(getattr(operator, op), types.Function(op_type))


OperatorModuleAttribute = builtin_attr(OperatorModuleAttribute)

builtin_global(operator, types.Module(operator))

