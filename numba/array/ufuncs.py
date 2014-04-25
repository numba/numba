from array import Array
from nodes import UnaryOperation, BinaryOperation


def create_unary_op(op_str):
    def unary_op(operand):
        return Array(data=UnaryOperation(operand.array_node, op_str))
    return unary_op

def create_binary_op(op_str):
    def binary_op(operand1, operand2):
        return Array(data=BinaryOperation(operand1.array_node, operand2.array_node, op_str))
    return binary_op


global_dict = globals()

unary_ufuncs = {'abs':'abs', 'log':'math.log'}

for name, op in unary_ufuncs.items():
    global_dict[name] = create_unary_op(op)

binary_ufuncs = {'add':'operator.add'}

for name, op in binary_ufuncs.items():
    global_dict[name] = create_binary_op(op)

