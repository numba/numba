from pyalge import Case, of
from nodes import *
import operator
import numpy as np


class Value(Case):

    @of('ArrayNode(data, owners)')
    def array_node(self, data, owners):
        return Value(data)

    @of('ArrayDataNode(array_data)')
    def array_data_node(self, array_data):
        return array_data

    @of('ScalarNode(value)')
    def scalar_node(self, value):
        return value

    @of('UnaryOperation(operand, op_str)')
    def unary_operation(self, operand, op_str):
        return eval(op_str)(Value(operand))

    @of('BinaryOperation(lhs, rhs, op_str)')
    def binary_operation(self, lhs, rhs, op_str):
        return eval(op_str)(Value(lhs), Value(rhs))

    @of('ArrayAssignOperation(operand, key, value)')
    def array_assign_operation(self, operand, key, value):
        operator.setitem(Value(operand), key, Value(value))
        return Value(operand)

    @of('WhereOperation(cond, left, right)')
    def where_operation(self, cond, left, right):
        return np.where(Value(cond), Value(left), Value(right))

