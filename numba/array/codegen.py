from __future__ import print_function, absolute_import
from .pyalge import Case, of
from .nodes import *
from numba import vectorize, typeof
import operator
import math
import sys
import numpy


# IEEE-754 guarantees that 17 decimal digits are enough to represent any
# double value in a string representation and convert back to original
# representation without loss of information.
FLOAT_EXACT_FMT = "%.17g"


class MissingArgumentError(Exception):

    def __init__(self, arg_name):
        self.arg_name = arg_name

    def __str__(self):
        return '{0} argument was not specified'.format(self.arg_name)


class CodeGen(Case):

    @of('ArrayNode(data, owners, depth)')
    def array(self, data, owners, depth):
        return CodeGen(data, state=self.state)

    @of('ArrayDataNode(array_data, depth)')
    def array_data_node(self, array_data, depth):
        input_ids = [id(x) for x in self.state['inputs']]
        if id(array_data) in input_ids:
            in_str = self.state['input_names'][input_ids.index(id(array_data))]
        else:
            in_str = 'in' + str(len(self.state['inputs']))
            self.state['inputs'].append(array_data)
            self.state['input_names'].append(in_str)
            self.state['input_types'].append(str(typeof(array_data).dtype))
        return in_str

    @of('VariableDataNode(name, depth)')
    def variable_data_node(self, name, depth):
        self.state['variable_found'] = True
        if name not in self.state['variables'].keys():
            raise MissingArgumentError(name)
        if name not in self.state['input_names']:
            data = self.state['variables'][name]
            self.state['input_names'].append(name)
            self.state['inputs'].append(data)
            self.state['input_types'].append(str(typeof(data).dtype))
            self.state['variable_names'].append(name)
        return name

    @of('ScalarNode(value, depth)')
    def scalar_node(self, value, depth):
        text = str(value)
        if isinstance(value, float) and type(value)(text) != text:
            return FLOAT_EXACT_FMT % value
        else:
            return text

    @of('UnaryOperation(operand, op_str, depth)')
    def unary_operation(self, operand, op_str, depth):
        operand_var = CodeGen(operand, state=self.state)
        temp_var = 'temp' + str(len(self.state['vectorize_body']))
        if op_str == 'square':
            line = '{0} = operator.pow({1}, 2)'.format(temp_var, operand_var)
        else:
            line = '{0} = numpy.{1}({2})'.format(temp_var,
                                                 op_str,
                                                 operand_var)
        self.state['vectorize_body'].append(line)
        return temp_var

    @of('BinaryOperation(lhs, rhs, op_str, depth)')
    def binary_operation(self, lhs, rhs, op_str, depth):
        lhs_var = CodeGen(lhs, state=self.state)
        rhs_var = CodeGen(rhs, state=self.state)
        temp_var = 'temp' + str(len(self.state['vectorize_body']))
        line = '{0} = numpy.{1}({2}, {3})'.format(temp_var,
                                                  op_str,
                                                  lhs_var,
                                                  rhs_var)
        self.state['vectorize_body'].append(line)
        return temp_var

    @of('WhereOperation(cond, left, right, depth)')
    def where_operation(self, cond, left, right, depth):
        cond_var = CodeGen(cond, state=self.state)
        left_var = CodeGen(left, state=self.state)
        right_var = CodeGen(right, state=self.state)
        temp_var = 'temp' + str(len(self.state['vectorize_body']))
        line = '{0} = {1} if {2} else {3}'.format(temp_var,
                                                  left_var,
                                                  cond_var,
                                                  right_var)
        self.state['vectorize_body'].append(line)
        return temp_var


def build(array, state):
    state['inputs'] = []
    state['input_names'] = []
    state['input_types'] = []
    state['vectorize_body'] = []
    state['variable_names'] = []
    output_var = CodeGen(array.array_node, state=state)
    #return (state['inputs'], state['input_names'], state['variable_names'],
    #        state['input_types'], state['vectorize_body'], output_var)
    state['output_var'] = output_var


vectorize_template = ('def foo({0}):\n'
                      '    {1}\n'
                      '    return {2}\n')


#def run(inputs, input_names, variable_names, input_types, vectorize_body, output_var):
def run(state):

    ufunc_str = vectorize_template.format(','.join(state['input_names']),
                                          '\n    '.join(state['vectorize_body']),
                                          state['output_var'])

    code = compile(ufunc_str, '<string>', 'exec')
    # JNB: better way to build globals?
    exec(code, globals())
    foo = globals()['foo']

    ufunc = vectorize('({0},)'.format(','.join(state['input_types'])), nopython=True)(foo)
    state['ufunc'] = ufunc
    return ufunc(*state['inputs'])


#def dump(inputs, input_names, variable_names, input_types, vectorize_body, output_var):
def dump(state):
    vectorize_str = vectorize_template.format(','.join(state['input_names']),
                                              '\n    '.join(state['vectorize_body']),
                                              state['output_var'])
    return '@vectorize(["({0},)"])\n'.format(','.join(state['input_types'])) + vectorize_str
    return decorator + ufunc_str
