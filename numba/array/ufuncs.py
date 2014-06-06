#from __future__ import division
from nodes import UnaryOperation, BinaryOperation, ScalarNode
import numba._npymath_exports as nbmath
from array import Array, reduce_


def create_unary_op(op_str):
    def unary_op(operand):
        return Array(data=UnaryOperation(operand.array_node, op_str))
    return unary_op

def create_binary_op(op_str):
    def binary_op(operand1, operand2):
        def parse_operand(operand):
            if isinstance(operand, Array):
                return operand.array_node
            elif isinstance(operand, (int, long, float)):
                return ScalarNode(operand)
            else:
                raise TypeError('Invalid type ({0})for binary operation'.format(type(operand)))
        operand1 = parse_operand(operand1)
        operand2 = parse_operand(operand2)
        if isinstance(operand1, ScalarNode) and isinstance(operand2, ScalarNode):
            scalar = Array(data=BinaryOperation(operand1, operand2, op_str)).eval(use_python=True)
            return Array(data=ScalarNode(scalar))
        return Array(data=BinaryOperation(operand1, operand2, op_str))
    return binary_op


global_dict = globals()

# to make the groupings like numpy mathematical functions documentaton separated
# the hyperbolic trig from regular trig functions 
# moved hypot and arctan2 to binary trig funcs
unary_trig_ufuncs = {'sin':'math.sin', 'cos':'math.cos', 'tan':'math.tan',
    'arcsin':'math.asin', 'arccos':'math.acos', 'arctan':'math.atan',
    'degrees':'math.degrees' ,'deg2rad':'math.radians', 'rad2deg':'math.degrees',
    'radians':'math.radians'}

for name, op in unary_trig_ufuncs.items():
    global_dict[name] = create_unary_op(op)

binary_trig_ufuncs = {'hypot':'math.hypot', 'arctan2':'math.atan2'}

for name, op in binary_trig_ufuncs.items():
    global_dict[name] = create_binary_op(op)

unary_hyperbolic_ufuncs = {'sinh':'math.sinh', 'cosh':'math.cosh',
    'tanh':'math.tanh', 'arcsinh':'math.asinh', 'arccosh':'math.acosh',
    'arctanh':'math.atanh'}

for name, op in unary_hyperbolic_ufuncs.items():
    global_dict[name] = create_unary_op(op)

# todo -- rint, around, round, fix (these are functions, not ufuncs according to numpy doc)
unary_rounding_ufuncs = {'floor':'math.floor', 'ceil':'math.ceil', 'trunc':'math.trunc', 'rint':'numpy.rint'}

for name, op in unary_rounding_ufuncs.items():
    global_dict[name] = create_unary_op(op)

#todo --  logaddep, logaddexp2
unary_exps_logs_ufuncs = {'exp':'math.exp', 'exp2':'numpy.exp2', 'expm1':'math.expm1', 'log':'math.log', 'log10':'math.log10', 'log1p':'math.log1p', 'log2':'numpy.log2'}

for name, op in unary_exps_logs_ufuncs.items():
    global_dict[name] = create_unary_op(op)

# todo logaddexp does not work
# need to add binary_ufunc capability to numba
binary_exps_logs_ufuncs = {'logaddexp':'numpy.logaddexp'}

for name, op in binary_exps_logs_ufuncs.items():
    global_dict[name] = create_binary_op(op)


# todo -- sign is broken
unary_misc_ufuncs = {'sqrt':'math.sqrt', 'fabs':'numpy.fabs'}

for name, op in unary_misc_ufuncs.items():
    global_dict[name] = create_unary_op(op)

# todo -- currently frexp fails and ldexp is a binary op
#unary_floating_point_ufuncs = {'frexp':'math.frexp', 'ldexp':'math.ldexp'}
#
#for name, op in unary_floating_point_ufuncs.items():
#    global_dict[name] = create_unary_op(op)
#



unary_arithmetic_ufuncs = {'negative':'operator.neg'}

for name, op in unary_arithmetic_ufuncs.items():
    global_dict[name] = create_unary_op(op)


binary_arithmetic_ufuncs = {'add':'operator.add', 'subtract':'operator.sub',
        'multiply':'operator.mul', 'power':'operator.pow', 'division':'operator.div', 'floor_division':'operator.floordiv'}

for name, op in binary_arithmetic_ufuncs.items():
    global_dict[name] = create_binary_op(op)

