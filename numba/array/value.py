from __future__ import print_function, absolute_import
from .pyalge import Case, of
from .nodes import *
import operator
import numpy


class Value(Case):

    @of('ArrayNode(data, owners, depth)')
    def array_node(self, data, owners, depth):
        return Value(data, state=self.state)

    @of('ArrayDataNode(array_data, depth)')
    def array_data_node(self, array_data, depth):
        return array_data

    @of('ScalarNode(value, depth)')
    def scalar_node(self, value, depth):
        return value

    @of('VariableDataNode(name, depth)')
    def variable_data_node(self, name, depth):
        if name not in self.state['variables'].keys():
            raise MissingArgumentError(name)
        return self.state['variables'][name]

    @of('UnaryOperation(operand, op_str, depth)')
    def unary_operation(self, operand, op_str, depth):
        op = getattr(numpy, op_str)
        return op(Value(operand))

    @of('BinaryOperation(lhs, rhs, op_str, depth)')
    def binary_operation(self, lhs, rhs, op_str, depth):
        op = getattr(numpy, op_str)
        return op(Value(lhs), Value(rhs))

    @of('ArrayAssignOperation(operand, key, value, depth)')
    def array_assign_operation(self, operand, key, value, depth):
        operator.setitem(Value(operand), key, Value(value))
        return Value(operand)

    @of('WhereOperation(cond, left, right, depth)')
    def where_operation(self, cond, left, right, depth):
        return numpy.where(Value(cond), Value(left), Value(right))

    @of('UFuncNode(ufunc, args, depth)')
    def ufunc_node(self, ufunc, args, depth):
        new_args = []
        for arg in args:
            new_args.append(Value(arg, state=self.state))
        return ufunc(*new_args)
