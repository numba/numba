import numpy as np
from numba import jit
import operator
import math
import weakref
from pyalge import datatype
import codegen
from value import Value
from repr_ import Repr
from nodes import *



def unary_op(op, op_str):

    def wrapper(func):
        def impl(self):
            return Array(data=UnaryOperation(self.array_node, op, op_str))

        return impl

    return wrapper


def binary_op(op, op_str):

    def wrapper(func):
        def impl(self, other):
            if isinstance(other, Array):
                other = other.array_node
            else:
                other = ScalarConstantNode(other)
            return Array(data=BinaryOperation(self.array_node, other, op, op_str))

        return impl

    return wrapper


class Array(object):

    def __init__(self, data=None):
        if isinstance(data, np.ndarray):
            data = ArrayDataNode(array_data=data)
        self._ref = weakref.ref(self)
        self.array_node = ArrayNode(data=data, owners=set([self._ref]))

    def __del__(self):
        self.array_node.owners.discard(self._ref)

    def eval(self, python=False):
        if not isinstance(self.array_node.data, ArrayDataNode):
            if python:
                data = Value(self.array_node)
            else:
                data = codegen.run(*codegen.build(self))
            self.array_node.data = ArrayDataNode(data)
        return self.array_node.data.array_data

    def __str__(self):
        return str(self.eval())

    def __repr__(self):
        return Repr(self.array_node, state={'level':0})

    @binary_op(operator.add, 'operator.add')
    def __add__(self, other):
        pass

    @binary_op(operator.sub, 'operator.sub')
    def __sub__(self, other):
        pass

    @binary_op(operator.mul, 'operator.mul')
    def __mul__(self, other):
        pass

    @binary_op(operator.div, 'operator.div')
    def __div__(self, other):
        pass

    @binary_op(operator.le, 'operator.le')
    def __le__(self, other):
        pass

    @binary_op(operator.pow, 'operator.pow')
    def __pow__(self, other):
        pass

    @binary_op(operator.getitem, 'operator.getitem')
    def __getitem__(self, other):
        pass

    def __setitem__(self, key, value):
        if isinstance(value, Array):
            value = value.array_node
        else:
            value = ScalarConstantNode(value)
        self.array_node = ArrayNode(data=ArrayAssignOperation(self.array_node, key, value),
                                    owners=set(self._ref))


def reduce_(func, operand, initial):
    array = operand.eval()

    cfunc = jit(func)
    
    @jit
    def reduce_loop(cfunc, array):
        total = 0
        for i in range(array.shape[0]):
            total = cfunc(total, array[i])
        return total

    return reduce_loop(cfunc, array)


def create_reduce_func(op, initial):
    
    def reduce_wrapper(operand):
        return reduce_(op, operand, initial)

    return reduce_wrapper


@unary_op(abs, 'abs')
def abs_(operand):
    pass

@unary_op(math.log, 'math.log')
def log(operand):
    pass

@binary_op(operator.add, 'operator.add')
def add(op1, op2):
    pass

add.reduce = create_reduce_func(lambda x,y: x + y, 0)


