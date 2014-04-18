from pyalge import Case, of
from nodes import *
from numba import vectorize
import operator


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

    @of('ScalarConstantNode(value)')
    def scalar_constant(self, value):
        return str(value)

    @of('UnaryOperation(operand, op, op_str)')
    def unary_operation(self, operand, op, op_str):
        return op_str + '(' + CodeGen(operand, state=self.state) + ')'

    @of('BinaryOperation(lhs, rhs, op, op_str)')
    def binary_operation(self, lhs, rhs, op, op_str):
        return op_str + '(' + CodeGen(lhs, state=self.state) + ',' + \
            CodeGen(rhs, state=self.state) + ')'


def build(array):
    inputs = []
    input_names = []
    input_types = []
    operations = CodeGen(array.array_node, state={'inputs':inputs,
                                                  'input_names':input_names,
                                                  'input_types':input_types})
    return operations, inputs, input_names, input_types


def run(operations, inputs, input_names, input_types):
    ufunc_str = '''
def foo({0}):
    return {1}
'''.format(','.join(input_names), operations)

    code = compile(ufunc_str, '<string>', 'exec')

    exec(code, globals())

    foo = globals()['foo']

    ufunc = vectorize('(' + ','.join(input_types) + ')')(foo)
    return ufunc(*inputs)
    
