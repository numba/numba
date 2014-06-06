#from __future__ import division
import numpy as np
from numba import jit
import math
import weakref
from pyalge import datatype
import codegen
from value import Value
from repr_ import Repr
from nodes import *
from numba.config import PYVERSION

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
        expected_args = ['use_python', 'debug']

        python = kwargs.get('use_python', False)
        debug = kwargs.get('debug', False)

        variables = dict([(key,value) for key,value in kwargs.items() if key not in expected_args])
        state = {'variables':variables, 'variable_found':False}

        if isinstance(self.array_node.data, ScalarNode):
            return Value(self.array_node, state=state)

        if not isinstance(self.array_node.data, ArrayDataNode):
            if debug:
                print codegen.dump(*codegen.build(self, state))

            if python:
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

    def __add__(self, other):
        return ufuncs.add(self, other)

    def __radd__(self, other):
        return ufuncs.add(self, other)

    def __iadd__(self, other):
        self.array_node = ufuncs.add(self, other).array_node
        return self

    def __sub__(self, other):
        return ufuncs.subtract(self, other)

    def __rsub__(self, other):
        return ufuncs.subtract(self, other)

    def __mul__(self, other):
        return ufuncs.multiply(self, other)

    def __rmul__(self, other):
        return ufuncs.multiply(self, other)

    def __div__(self, other):
        return ufuncs.division(self, other)

    def __rdiv__(self, other):
        return ufuncs.division(self, other)

    def __truediv__(self, other):
        return ufuncs.division(self, other)

    def __rtruediv__(self, other):
        return ufuncs.division(self, other)

    def __floordiv__(self, other):
        return ufuncs.floor_division(self, other)

    def __rfloordiv__(self, other):
        return ufuncs.floor_division(self, other)

    def __neg__(self):
        return ufuncs.negative(self)

    def __pow__(self, other):
        return ufuncs.power(self, other)

    def __rpow__(self, other):
        return ufuncs.power(self, other)

    def __getitem__(self, other):
        return Array(data=self.eval()[other])

    def __setitem__(self, key, value):
        data = self.eval()
        data[key] = value.eval()
        return Array(data=data)

    def __gt__(self, other):
        return ufuncs.greater(self, other)

    def sum(self, *args, **kwargs):
        result = self.eval(debug=False)
        sum_result = result.sum()
        return sum_result


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


import ufuncs
