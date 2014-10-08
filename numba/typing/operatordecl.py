import operator
from numba import types
from numba import utils
from numba.typing.templates import (AttributeTemplate, ConcreteTemplate,
                                    signature, Registry)

registry = Registry()
builtin_attr = registry.register_attr
builtin_global = registry.register_global


class TruthOperator(ConcreteTemplate):
    cases = [signature(types.boolean, op) for op in sorted(types.signed_domain)]
    cases += [signature(types.boolean, op) for op in sorted(types.unsigned_domain)]
    cases += [signature(types.boolean, op) for op in sorted(types.real_domain)]
    cases += [signature(types.boolean, op) for op in sorted(types.complex_domain)]

class UnaryOperator(ConcreteTemplate):
    cases = [signature(op, op) for op in sorted(types.signed_domain)]
    cases += [signature(op, op) for op in sorted(types.unsigned_domain)]
    cases += [signature(op, op) for op in sorted(types.real_domain)]
    cases += [signature(op, op) for op in sorted(types.complex_domain)]

class BinaryOperator(ConcreteTemplate):
    cases = [signature(op, op, op) for op in sorted(types.signed_domain)]
    cases += [signature(op, op, op) for op in sorted(types.unsigned_domain)]
    cases += [signature(op, op, op) for op in sorted(types.real_domain)]
    cases += [signature(op, op, op) for op in sorted(types.complex_domain)]

class TruedivOperator(ConcreteTemplate):
    cases = [signature(types.float64, op, op) for op in sorted(types.signed_domain)]
    cases += [signature(types.float64, op, op) for op in sorted(types.unsigned_domain)]
    cases += [signature(op, op, op) for op in sorted(types.real_domain)]
    cases += [signature(op, op, op) for op in sorted(types.complex_domain)]

class PowerOperator(ConcreteTemplate):
    cases = [signature(types.float64, types.float64, op)
             for op in sorted(types.signed_domain)]
    cases += [signature(types.float64, types.float64, op)
              for op in sorted(types.unsigned_domain)]
    cases += [signature(op, op, op) for op in sorted(types.real_domain)]
    cases += [signature(op, op, op) for op in sorted(types.complex_domain)]

class ComparisonOperator(ConcreteTemplate):
    cases = [signature(types.boolean, op, op) for op in sorted(types.signed_domain)]
    cases += [signature(types.boolean, op, op) for op in sorted(types.unsigned_domain)]
    cases += [signature(types.boolean, op, op) for op in sorted(types.real_domain)]
    cases += [signature(types.boolean, op, op) for op in sorted(types.complex_domain)]


unary_operators = ['pos', 'neg', 'invert']
truth_operators = ['not_']

for op in unary_operators:
    op_type = type('Operator_' + op, (UnaryOperator,), {'key':getattr(operator, op)})
    builtin_global(getattr(operator, op), types.Function(op_type))

for op in truth_operators:
    op_type = type('Operator_' + op, (TruthOperator,), {'key':getattr(operator, op)})
    builtin_global(getattr(operator, op), types.Function(op_type))


binary_operators = ['add', 'sub', 'mul', 'div', 'floordiv', 'mod',
                    'and_', 'or_', 'xor', 'lshift', 'rshift',
                    'iadd', 'isub', 'imul', 'idiv', 'ifloordiv', 'imod',
                    'iand', 'ior', 'ixor', 'ilshift', 'irshift',
                    ]
comparison_operators = ['eq', 'ne', 'lt', 'le', 'gt', 'ge']
truediv_operators = ['truediv', 'itruediv']
power_operators = ['pow', 'ipow']

if utils.IS_PY3:
    binary_operators.remove('div')
    binary_operators.remove('idiv')

for op in binary_operators:
    op_type = type('Operator_' + op, (BinaryOperator,), {'key':getattr(operator, op)})
    builtin_global(getattr(operator, op), types.Function(op_type))

for op in comparison_operators:
    op_type = type('Operator_' + op, (ComparisonOperator,), {'key':getattr(operator, op)})
    builtin_global(getattr(operator, op), types.Function(op_type))

for op in truediv_operators:
    op_type = type('Operator_' + op, (TruedivOperator,), {'key':getattr(operator, op)})
    builtin_global(getattr(operator, op), types.Function(op_type))

for op in power_operators:
    op_type = type('Operator_' + op, (PowerOperator,), {'key':getattr(operator, op)})
    builtin_global(getattr(operator, op), types.Function(op_type))


builtin_global(operator, types.Module(operator))
