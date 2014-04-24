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
from numba.config import PYVERSION


# Wrapper for unary functions so that calling unary function
# will return Array object to user instead of UnaryOperation node
def unary_op(op_str):

    def wrapper(func):
        def impl(self):
            return Array(data=UnaryOperation(self.array_node, op_str))

        return impl

    return wrapper


# Wrapper for binary functions so that calling binary function
# will return Array object to user instead of BinaryOperation node.
def binary_op(op_str):

    def wrapper(func):
        def impl(self, other):
            if isinstance(other, Array):
                other = other.array_node
            else:
                other = ScalarConstantNode(other)
            return Array(data=BinaryOperation(self.array_node, other, op_str))
        return impl

    return wrapper


class Array(object):

    def __init__(self, data=None, name=None):
        if isinstance(data, np.ndarray):
            data = ArrayDataNode(array_data=data)
        self._ref = weakref.ref(self)
        if data is None:
            self.array_node = VariableDataNode(name=name)
        else:
            self.array_node = ArrayNode(data=data, owners=set([self._ref]))

    def __del__(self):
        if isinstance(self.array_node, ArrayNode):
            self.array_node.owners.discard(self._ref)

    def eval(self, **kwargs):
        expected_args = ['python', 'debug']

        python = kwargs.get('python', False)
        debug = kwargs.get('debug', False)
        
        variables = dict([(key,value) for key,value in kwargs.items() if key not in expected_args])
        state = {'variables':variables, 'variable_found':False}

        if not isinstance(self.array_node.data, ArrayDataNode):
            if debug:
                return codegen.dump(*codegen.build(self, state))
            elif python:
                data = Value(self.array_node, state=state)
            else:
                data = codegen.run(*codegen.build(self, state))
            if not state['variable_found']:
                self.array_node.data = ArrayDataNode(array_data=data)
            else:
                return data
        return self.array_node.data.array_data

    def __str__(self):
        return str(self.eval())

    def __repr__(self):
        return Repr(self.array_node, state={'level':0})

    @binary_op('operator.add')
    def __add__(self, other):
        pass

    @binary_op('operator.sub')
    def __sub__(self, other):
        pass

    @binary_op('operator.mul')
    def __mul__(self, other):
        pass

    @binary_op('operator.truediv' if PYVERSION >= (3, 0) else 'operator.div')
    def __div__(self, other):
        pass

    @binary_op('operator.lt')
    def __lt__(self, other):
        pass

    @binary_op('operator.le')
    def __le__(self, other):
        pass

    @binary_op('operator.gt')
    def __gt__(self, other):
        pass

    @binary_op('operator.ge')
    def __ge__(self, other):
        pass

    @binary_op('operator.eq')
    def __eq__(self, other):
        pass

    @binary_op('operator.ne')
    def __ne__(self, other):
        pass

    @binary_op('operator.pow')
    def __pow__(self, other):
        pass

    def __getitem__(self, other):
        return Array(data=self.eval()[other])

    def __setitem__(self, key, value):
        data = self.eval()
        data[key] = value.eval()
        return Array(data=data)


Array_methods = ['min', 'max', 'any', 'all']

for method in Array_methods:
    def create_method_template(method):
        def method_template(self, *args, **kwargs):
            return getattr(self.eval(), method)(*args, **kwargs)
        return method_template
    setattr(Array, method, create_method_template(method))


def reduce_(func, operand, initial_value):
    array = operand.eval()

    cfunc = jit(func)
    
    @jit
    def reduce_loop(cfunc, array, initial_value):
        total = 0
        for i in range(array.shape[0]):
            total = cfunc(total, array[i])
        return total

    return reduce_loop(cfunc, array, initial_value)


def create_reduce_func(operation, initial_value):
    
    def reduce_wrapper(operand):
        return reduce_(operation, operand, initial_value)

    return reduce_wrapper


@unary_op('abs')
def abs_(operand):
    pass

@unary_op('math.log')
def log(operand):
    pass

@binary_op('operator.add')
def add(operand1, operand2):
    pass

add.reduce = create_reduce_func(lambda x,y: x+y, 0)


