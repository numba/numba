from __future__ import print_function, absolute_import
from .pyalge import Case, of
from .nodes import *
import operator
import numpy


class Value(Case):

    @of('ArrayNode(data, owners)')
    def array_node(self, data, owners):
        return Value(data, state=self.state)

    @of('ArrayDataNode(array_data)')
    def array_data_node(self, array_data):
        return array_data

    @of('ScalarNode(value)')
    def scalar_node(self, value):
        return value

    @of('VariableDataNode(name)')
    def variable_data_node(self, name):
        if name not in self.state['variables'].keys():
            raise MissingArgumentError(name)
        return self.state['variables'][name]

    @of('UnaryOperation(operand, op_str)')
    def unary_operation(self, operand, op_str):
        op = getattr(numpy, op_str)
        return op(Value(operand))

    @of('BinaryOperation(lhs, rhs, op_str)')
    def binary_operation(self, lhs, rhs, op_str):
        op = getattr(numpy, op_str)
        return op(Value(lhs), Value(rhs))

    @of('ArrayAssignOperation(operand, key, value)')
    def array_assign_operation(self, operand, key, value):
        operator.setitem(Value(operand), key, Value(value))
        return Value(operand)

    @of('WhereOperation(cond, left, right)')
    def where_operation(self, cond, left, right):
        return numpy.where(Value(cond), Value(left), Value(right))

    @of('UFuncNode(ufunc, args)')
    def ufunc_node(self, ufunc, args):
        new_args = []
        for arg in args:
            new_args.append(Value(arg, state=self.state))
        return ufunc(*new_args)
