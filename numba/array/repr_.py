from pyalge import Case, of
from nodes import *


def get_indent(level):
    return ''.join([' '] * level * 4)

class Repr(Case):

    @of('ArrayNode(data, owners)')
    def array_node(self, data, owners):
        level = self.state['level']
        return '{0}ArrayNode (owned={1}): \n{2}'.format(get_indent(level),
            str(bool(owners)),
            str(Repr(data, state={'level':level+1})))

    @of('ArrayDataNode(array_data)')
    def array_data_node(self, array_data):
        level = self.state['level']
        return '{0}array_data: {1}\n'.format(get_indent(level), str(array_data))

    @of('VariableDataNode(name)')
    def variable_data_node(self, name):
        level = self.state['level']
        return '{0}variable_data: {1}\n'.format(get_indent(level), name)

    @of('ScalarConstantNode(value)')
    def scalar_constant(self, value):
        level = self.state['level']
        return '{0}value: {1}\n'.format(get_indent(level), str(value))

    @of('UnaryOperation(operand, op, op_str)')
    def unary_operation(self, operand, op, op_str):
        level = self.state['level']
        return '{0}UnaryOperation: \n{1}\n'.format(get_indent(level),
            Repr(operand, state={'level':level+1}))

    @of('BinaryOperation(lhs, rhs, op, op_str)')
    def binary_operation(self, lhs, rhs, op, op_str):
        level = self.state['level']
        return '{0}BinaryOperation: \n{1}\n{2}\n'.format(get_indent(level),
            Repr(lhs, state={'level':level+1}),
            Repr(rhs, state={'level':level+1}))

    @of('ArrayAssignOperation(operand, key, value)')
    def array_assign_operation(self, operand, key, value):
        level = self.state['level']
        return '{0}ArrayAssignOperation: \n{1}\n{2}\n'.format(get_indent(level),
            Repr(key, state={'level':level+1}),
            Repr(value, state={'level':level+1}))

