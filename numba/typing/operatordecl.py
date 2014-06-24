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

class TruthOperator(ConcreteTemplate):
    cases = [
        signature(types.bool_, types.int64),
        signature(types.bool_, types.uint64),
        signature(types.bool_, types.float32),
        signature(types.bool_, types.float64),
        signature(types.bool_, types.complex64),
        signature(types.bool_, types.complex128),
    ]

class UnaryOperator(ConcreteTemplate):
    cases = [
        # Required for operator.invert to work correctly on 32-bit ints
        # (otherwise a 64-bit result with 32 high 1s would be returned)
        signature(types.int32, types.int32),
        signature(types.uint32, types.uint32),
        signature(types.int64, types.int64),
        signature(types.uint64, types.uint64),
        signature(types.float32, types.float32),
        signature(types.float64, types.float64),
        signature(types.complex64, types.complex64),
        signature(types.complex128, types.complex128),
    ]

class BinaryOperator(ConcreteTemplate):
    cases = [
        signature(types.int64, types.int64, types.int64),
        signature(types.uint64, types.uint64, types.uint64),
        signature(types.float32, types.float32, types.float32),
        signature(types.float64, types.float64, types.float64),
        signature(types.complex64, types.complex64, types.complex64),
        signature(types.complex128, types.complex128, types.complex128),
    ]

class TruedivOperator(ConcreteTemplate):
    cases = [
        signature(types.float64, types.int64, types.int64),
        signature(types.float64, types.uint64, types.uint64),
        signature(types.float32, types.float32, types.float32),
        signature(types.float64, types.float64, types.float64),
        signature(types.complex64, types.complex64, types.complex64),
        signature(types.complex128, types.complex128, types.complex128),
    ]

class PowerOperator(ConcreteTemplate):
    cases = [
        signature(types.float64, types.float64, types.int64),
        signature(types.float64, types.float64, types.uint64),
        signature(types.float32, types.float32, types.float32),
        signature(types.float64, types.float64, types.float64),
        signature(types.complex64, types.complex64, types.complex64),
        signature(types.complex128, types.complex128, types.complex128),
    ]

class ComparisonOperator(ConcreteTemplate):
    cases = [
        signature(types.bool_, types.int64, types.int64),
        signature(types.bool_, types.uint64, types.uint64),
        signature(types.bool_, types.float32, types.float32),
        signature(types.bool_, types.float64, types.float64),
        signature(types.bool_, types.complex64, types.complex64),
        signature(types.bool_, types.complex128, types.complex128),
    ]


def create_resolve_method(op):
    def resolve_op(self, mod):
        return types.Function(op)
    return resolve_op


unary_operators = ['pos', 'neg', 'invert']
truth_operators = ['not_']

for op in unary_operators:
    op_type = type('Operator_' + op, (UnaryOperator,), {'key':getattr(operator, op)})
    setattr(OperatorModuleAttribute, 'resolve_' + op,  create_resolve_method(op_type))
    builtin_global(getattr(operator, op), types.Function(op_type))

for op in truth_operators:
    op_type = type('Operator_' + op, (TruthOperator,), {'key':getattr(operator, op)})
    setattr(OperatorModuleAttribute, 'resolve_' + op,  create_resolve_method(op_type))
    builtin_global(getattr(operator, op), types.Function(op_type))


binary_operators = ['add', 'sub', 'mul', 'div', 'floordiv', 'mod',
                    'and_', 'or_', 'xor', 'lshift', 'rshift']
comparison_operators = ['eq', 'ne', 'lt', 'le', 'gt', 'ge']
truediv_operators = ['truediv']
power_operators = ['pow']

if utils.IS_PY3:
    binary_operators.remove('div')
    
for op in binary_operators:
    op_type = type('Operator_' + op, (BinaryOperator,), {'key':getattr(operator, op)})
    setattr(OperatorModuleAttribute, 'resolve_' + op,  create_resolve_method(op_type))
    builtin_global(getattr(operator, op), types.Function(op_type))

for op in comparison_operators:
    op_type = type('Operator_' + op, (ComparisonOperator,), {'key':getattr(operator, op)})
    setattr(OperatorModuleAttribute, 'resolve_' + op,  create_resolve_method(op_type))
    builtin_global(getattr(operator, op), types.Function(op_type))

for op in truediv_operators:
    op_type = type('Operator_' + op, (TruedivOperator,), {'key':getattr(operator, op)})
    setattr(OperatorModuleAttribute, 'resolve_' + op,  create_resolve_method(op_type))
    builtin_global(getattr(operator, op), types.Function(op_type))

for op in power_operators:
    op_type = type('Operator_' + op, (PowerOperator,), {'key':getattr(operator, op)})
    setattr(OperatorModuleAttribute, 'resolve_' + op,  create_resolve_method(op_type))
    builtin_global(getattr(operator, op), types.Function(op_type))


OperatorModuleAttribute = builtin_attr(OperatorModuleAttribute)

builtin_global(operator, types.Module(operator))

