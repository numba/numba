from pyalge import Case, of
from nodes import *
import operator
import numpy


op_implementation = {
    'sin':'numpy.sin',
    'cos':'numpy.cos',
    'tan':'numpy.tan',
    'arcsin':'numpy.asin',
    'arccos':'numpy.acos',
    'arctan':'numpy.atan',
    'degrees':'numpy.degrees',
    'deg2rad':'numpy.radians',
    'rad2deg':'numpy.degrees',
    'radians':'numpy.radians',
    'sinh':'numpy.sinh',
    'cosh':'numpy.cosh',
    'tanh':'numpy.tanh',
    'arcsinh':'numpy.asinh',
    'arccosh':'numpy.acosh',
    'arctanh':'numpy.atanh',
    'floor':'numpy.floor',
    'ceil':'numpy.ceil',
    'trunc':'numpy.trunc',
    'rint':'numpy.rint',
    'exp':'numpy.exp',
    'exp2':'numpy.exp2',
    'expm1':'numpy.expm1',
    'log':'numpy.log',
    'log10':'numpy.log10',
    'log1p':'numpy.log1p',
    'log2':'numpy.log2',
    'sqrt':'numpy.sqrt',
    'fabs':'numpy.fabs',
    'abs':'numpy.abs',
    'negative':'operator.neg',
    'hypot':'numpy.hypot',
    'arctan2':'numpy.atan2',
    'logaddexp':'numpy.logaddexp',
    'add':'operator.add',
    'subtract':'operator.sub',
    'multiply':'operator.mul',
    'power':'operator.pow',
    'division':'operator.div',
    'floor_division':'operator.floordiv',
    'greater':'operator.gt',
}


class Value(Case):

    @of('ArrayNode(data, owners)')
    def array_node(self, data, owners):
        return Value(data)

    @of('ArrayDataNode(array_data)')
    def array_data_node(self, array_data):
        return array_data

    @of('ScalarNode(value)')
    def scalar_node(self, value):
        return value

    @of('UnaryOperation(operand, op_str)')
    def unary_operation(self, operand, op_str):
        return eval(op_implementation[op_str])(Value(operand))

    @of('BinaryOperation(lhs, rhs, op_str)')
    def binary_operation(self, lhs, rhs, op_str):
        return eval(op_implementation[op_str])(Value(lhs), Value(rhs))

    @of('ArrayAssignOperation(operand, key, value)')
    def array_assign_operation(self, operand, key, value):
        operator.setitem(Value(operand), key, Value(value))
        return Value(operand)

    @of('WhereOperation(cond, left, right)')
    def where_operation(self, cond, left, right):
        return numpy.where(Value(cond), Value(left), Value(right))

