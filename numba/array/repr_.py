from __future__ import print_function, absolute_import
from .pyalge import Case, of
from .nodes import *


def get_indent(level):
    return ''.join([' '] * level * 4)

class Repr(Case):

    @of('ArrayNode(data, owners, depth)')
    def array_node(self, data, owners, depth):
        level = self.state['level']
        return '{0}ArrayNode: \n{1}'.format(get_indent(level),
            str(Repr(data, state={'level':level+1})))

    @of('ArrayDataNode(array_data, depth)')
    def array_data_node(self, array_data, depth):
        level = self.state['level']
        return '{0}array_data: {1}\n'.format(get_indent(level), str(array_data))

    @of('VariableDataNode(name, depth)')
    def variable_data_node(self, name, depth):
        level = self.state['level']
        return '{0}variable_data: {1}\n'.format(get_indent(level), name)

    @of('ScalarNode(value, depth)')
    def scalar_node(self, value, depth):
        level = self.state['level']
        return '{0}ScalarNode: {1}\n'.format(get_indent(level), str(value))

    @of('UnaryOperation(operand, op_str, depth)')
    def unary_operation(self, operand, op_str, depth):
        level = self.state['level']
        return '{0}UnaryOperation: {1}\n{2}\n'.format(get_indent(level),
            op_str,
            Repr(operand, state={'level':level+1}))

    @of('BinaryOperation(lhs, rhs, op_str, depth)')
    def binary_operation(self, lhs, rhs, op_str, depth):
        level = self.state['level']
        return '{0}BinaryOperation: {1}\n{2}\n{3}\n'.format(get_indent(level),
            op_str,
            Repr(lhs, state={'level':level+1}),
            Repr(rhs, state={'level':level+1}))

    @of('ArrayAssignOperation(operand, key, value, depth)')
    def array_assign_operation(self, operand, key, value, depth):
        level = self.state['level']
        return '{0}ArrayAssignOperation: \n{1}\n{2}\n'.format(get_indent(level),
            Repr(key, state={'level':level+1}),
            Repr(value, state={'level':level+1}))

    @of('WhereOperation(cond, left, right, depth)')
    def where_operation(self, cond, left, right, depth):
        level = self.state['level']
        return '{0}WhereOperation: \n{1}\n{2}\n{3}\n'.format(get_indent(level),
            Repr(cond, state={'level':level+1}),
            Repr(left, state={'level':level+1}),
            Repr(right, state={'level':level+1}))

    @of('UFuncNode(ufunc, args, depth)')
    def ufunc_node(self, ufunc, args, depth):
        level = self.state['level']
        ufunc_str = '{0}UFuncNode: \n'.format(get_indent(level))
        for arg in args:
            ufunc_str = '{0}{1}\n'.format(ufunc_str, Repr(arg, state={'level':level+1}))
        return ufunc_str
