from nodes import UnaryOperation, BinaryOperation, ScalarNode
from array import Array


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

unary_trig_ufuncs = {'sin':'math.sin', 'cos':'math.cos', 'tan':'math.tan',
    'arcsin':'math.asin', 'arccos':'math.acos', 'arctan':'math.atan',
    'hypot':'math.hypot', 'arctan2':'math.atan2', 'degrees':'math.degrees',
    'radians':'math.radians', 'sinh':'math.sinh', 'cosh':'math.cosh',
    'tanh':'math.tanh', 'arcsinh':'math.asinh', 'arccosh':'math.acosh',
    'arctanh':'math.atanh'}

for name, op in unary_trig_ufuncs.items():
    global_dict[name] = create_unary_op(op)

unary_arithmetic_ufuncs = {'negative':'operator.neg'}

for name, op in unary_arithmetic_ufuncs.items():
    global_dict[name] = create_unary_op(op)


binary_arithmetic_ufuncs = {'add':'operator.add', 'subtract':'operator.sub',
    'multiply':'operator.mul', 'power':'operator.pow'}

for name, op in binary_arithmetic_ufuncs.items():
    global_dict[name] = create_binary_op(op)

