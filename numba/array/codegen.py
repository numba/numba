from pyalge import Case, of
from nodes import *
from numba import vectorize
import operator
import math
import sys


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
        var_str = 'x' + str(id(array_data))
        if not var_str in self.state['input_names']:
            self.state['inputs'].append(array_data)
            self.state['input_names'].append(var_str)
            self.state['input_types'].append(str(array_data.dtype))
        return var_str

    @of('VariableDataNode(name)')
    def variable_data_node(self, name):
        data = self.state['variables'][name]
        var_str = 'x' + str(id(data))
        self.state['inputs'].append(data)
        self.state['input_names'].append(var_str)
        self.state['input_types'].append(str(data.dtype))
        self.state['variable_found'] = True
        return var_str

    @of('ScalarNode(value)')
    def scalar_node(self, value):
        text = str(value)
        if isinstance(value, float) and type(value)(text) != text:
            return FLOAT_EXACT_FMT % value
        else:
            return text

    @of('UnaryOperation(operand, op_str)')
    def unary_operation(self, operand, op_str):
        return op_str + '(' + CodeGen(operand, state=self.state) + ')'

    @of('BinaryOperation(lhs, rhs, op_str)')
    def binary_operation(self, lhs, rhs, op_str):
        return op_str + '(' + CodeGen(lhs, state=self.state) + ',' + \
            CodeGen(rhs, state=self.state) + ')'

def build(array, state):
    state['inputs'] = []
    state['input_names'] = []
    state['input_types'] = []
    operations = CodeGen(array.array_node, state=state)
    return operations, state['inputs'], state['input_names'], state['input_types']


def run(operations, inputs, input_names, input_types):
    ufunc_str = '''
def foo({0}):
    return {1}
'''.format(','.join(input_names), operations)

    code = compile(ufunc_str, '<string>', 'exec')

    exec(code, globals())

    foo = globals()['foo']

    if len(input_types) > 1:
        ufunc = vectorize('(' + ','.join(input_types) + ')')(foo)
    else:
        ufunc = vectorize('(' + input_types[0] + ',)')(foo)
    return ufunc(*inputs)


def dump(operations, inputs, input_names, input_types):
    ufunc_str = '''
def foo({0}):
    return {1}
'''.format(','.join(input_names), operations)

    if len(input_types) > 1:
        decorator =  "@vectorize(['(" + ','.join(input_types) + ")']"
    else:
        decorator =  "@vectorize(['(" + input_types[0] + ",)']"

    return decorator + ufunc_str
