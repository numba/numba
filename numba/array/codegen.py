from pyalge import Case, of
from nodes import *
from numba import vectorize, typeof
import operator
import math
import sys
import numpy


# IEEE-754 guarantees that 17 decimal digits are enough to represent any
# double value in a string representation and convert back to original
# representation without loss of information.
FLOAT_EXACT_FMT = "%.17g"

class CodeGen(Case):

    @of('ArrayNode(data, owners)')
    def array(self, data, owners):
        return CodeGen(data, state=self.state)

    @of('ArrayDataNode(array_data)')
    def array_data_node(self, array_data):
        in_str = 'in' + str(len(self.state['inputs']))
        if not in_str in self.state['input_names']:
            self.state['inputs'].append(array_data)
            self.state['input_names'].append(in_str)
            self.state['input_types'].append(str(typeof(array_data).dtype))
        return in_str

    @of('VariableDataNode(name)')
    def variable_data_node(self, name):
        data = self.state['variables'][name]
        in_str = 'in' + str(len(self.state['inputs']))
        self.state['inputs'].append(data)
        self.state['input_names'].append(in_str)
        self.state['input_types'].append(str(typeof(data).dtype))
        self.state['variable_found'] = True
        return in_str

    @of('ScalarNode(value)')
    def scalar_node(self, value):
        text = str(value)
        if isinstance(value, float) and type(value)(text) != text:
            return FLOAT_EXACT_FMT % value
        else:
            return text

    @of('UnaryOperation(operand, op_str)')
    def unary_operation(self, operand, op_str):
        operand_var = CodeGen(operand, state=self.state)
        temp_var = 'temp' + str(len(self.state['vectorize_body']))
        self.state['vectorize_body'].append('{0} = {1}({2})'.format(temp_var, op_str, operand_var))
        return temp_var

    @of('BinaryOperation(lhs, rhs, op_str)')
    def binary_operation(self, lhs, rhs, op_str):
        lhs_var = CodeGen(lhs, state=self.state)
        rhs_var = CodeGen(rhs, state=self.state)
        temp_var = 'temp' + str(len(self.state['vectorize_body']))
        self.state['vectorize_body'].append('{0} = {1}({2}, {3})'.format(temp_var, op_str, lhs_var, rhs_var))
        return temp_var

    @of('WhereOperation(cond, left, right)')
    def where_operation(self, cond, left, right):
        return '({0} if {1} else {2})'.format(CodeGen(left, state=self.state),
                                             CodeGen(cond, state=self.state),
                                             CodeGen(right, state=self.state))
        cond_var = CodeGen(cond, state=self.state)
        left_var = CodeGen(left, state=self.state)
        right_var = CodeGen(right, state=self.state)
        temp_var = 'temp' + str(len(self.state['vectorize_body']))
        self.state['vectorize_body'].append('{0} = {1} if {2} else {3}'.format(temp_var, left_var, cond_var, right_var))
        return temp_var


def build(array, state):
    state['inputs'] = []
    state['input_names'] = []
    state['input_types'] = []
    state['vectorize_body'] = []
    output_var = CodeGen(array.array_node, state=state)
    return state['inputs'], state['input_names'], state['input_types'], state['vectorize_body'], output_var


vectorize_template = ('def foo({0}):\n'
                      '    {1}\n'
                      '    return {2}\n')


def run(inputs, input_names, input_types, vectorize_body, output_var):

    ufunc_str = vectorize_template.format(','.join(input_names),
                                          '\n    '.join(vectorize_body),
                                          output_var)

    code = compile(ufunc_str, '<string>', 'exec')
    exec(code, globals())
    foo = globals()['foo']

    ufunc = vectorize('({0},)'.format(','.join(input_types)))(foo)
    return ufunc(*inputs)


def dump(inputs, input_names, input_types, vectorize_body, output_var):
    vectorize_str = vectorize_template.format(','.join(input_names),
                                              '\n    '.join(vectorize_body),
                                              output_var)
    return '@vectorize(["({0},)"])\n'.format(','.join(input_types)) + vectorize_str

