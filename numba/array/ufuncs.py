from __future__ import print_function, absolute_import
from .nodes import UnaryOperation, BinaryOperation, ScalarNode, ArrayDataNode
from .array import Array, reduce_
import numpy


def create_unary_op(op_str):
    def unary_op(operand):
        if isinstance(operand, Array):
            depth = max(operand1.depth, operand2.depth)
            return Array(data=UnaryOperation(operand.array_node, op_str, depth))
        # JNB: do for binary ufuncs too
        elif isinstance(operand, numpy.ndarray):
            return Array(data=ArrayDataNode(getattr(numpy, op_str)(operand), 0))
        else:
            return Array(data=ScalarNode(getattr(numpy, op_str)(operand), 0))
    return unary_op

def create_binary_op(op_str):
    def binary_op(operand1, operand2):
        def parse_operand(operand):
            if isinstance(operand, Array):
                return operand.array_node
            elif isinstance(operand, (int, long, float)):
                return ScalarNode(operand, 0)
            else:
                raise TypeError('Invalid type ({0})for binary operation'.format(type(operand)))
        operand1 = parse_operand(operand1)
        operand2 = parse_operand(operand2)
        if isinstance(operand1, ScalarNode) and isinstance(operand2, ScalarNode):
            depth = max(operand1.depth, operand2.depth)
            temp_array = Array(data=BinaryOperation(operand1, operand2, op_str, depth))
            scalar = scalar.eval(use_python=True)
            return Array(data=ScalarNode(scalar))
        depth = max(operand1.depth, operand2.depth)
        return Array(data=BinaryOperation(operand1, operand2, op_str, depth))
    return binary_op


global_dict = globals()


# todo -- rint, around, round, fix (these are functions, not ufuncs according to numpy doc)
# todo --  logaddep, logaddexp2
# todo -- sign is broken
# todo -- currently frexp fails and ldexp is a binary op
# unary_floating_point_ufuncs = {'frexp':'math.frexp', 'ldexp':'math.ldexp'}
unary_ufuncs = ['sin', 'cos', 'tan', 'arcsin', 'arccos', 'arctan', 'degrees',
                'deg2rad', 'rad2deg', 'radians', 'sinh', 'cosh', 'tanh',
                'arcsinh', 'arccosh', 'arctanh', 'floor', 'ceil', 'trunc',
                'rint', 'exp', 'exp2', 'expm1', 'log', 'log10', 'log1p', 'log2',
                'sqrt', 'fabs', 'abs', 'negative', 'square', 'sign', 'signbit',
                'reciprocal', 'modf', 'logical_not', 'isnan', 'isfinite',
                'isinf', 'invert', 'frexp', 'conj', 'absolute'] 
                
for ufunc in unary_ufuncs:
    global_dict[ufunc] = create_unary_op(ufunc)


# todo logaddexp does not work
# need to add binary_ufunc capability to numba
binary_ufuncs = ['hypot', 'arctan2', 'logaddexp', 'add', 'subtract', 'multiply',
                 'power', 'divide', 'floor_divide', 'greater',
                 'true_divide', 'right_shift', 'remainder', 'not_equal',
                 'nextafter', 'mod', 'minimum', 'maximum', 'logical_xor',
                 'logical_or', 'logical_and', 'logaddexp2', 'less', 'less_equal',
                 'left_shift', 'ldexp', 'greater_equal', 'fmod', 'fmin', 'fmax',
                 'copysign', 'bitwise_xor', 'bitwise_or', 'bitwise_and']

for ufunc in binary_ufuncs:
    global_dict[ufunc] = create_binary_op(ufunc)

