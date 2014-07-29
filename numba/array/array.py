from __future__ import print_function, absolute_import
import numpy as np
from numba import jit
import math
import weakref
from .pyalge import datatype
from . import codegen
from .value import Value
from .repr_ import Repr
from .nodes import *
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

        # dict for state data when parsing expression graph
        state = {}

        # When parsing the array expression graph, and a variable array node
        # is found, the corresponding concrete array in variables dict will
        # be used in vectorize function.
        variables = dict([(key,value) for key,value in kwargs.items() if key not in expected_args])
        state['variables'] = variables

        # If a variable array node is found when parsing expression graph,
        # this will be set to True
        state['variable_found'] = False

        # JNB: cases where vectorize function is not needed?
        if isinstance(self.array_node.data, ScalarNode):
            return Value(self.array_node, state=state)
        elif isinstance(self.array_node.data, ArrayDataNode):
            return self.array_node.data.array_data
        elif isinstance(self.array_node.data, UFuncNode):
            return Value(self.array_node, state=state)
        elif python:
            return Value(self.array_node, state=state)
            
        # JNB: rename state variable to build_data
        # JNB: combine codegen build, dump, and run functions with state data
        # into Builder class
        codegen.build(self, state)
        if debug:
            # JNB: import __future__
            print(codegen.dump(state))
        result = codegen.run(state)
        
        # If variable arrays exist, save compiled vectorize function and
        # input args as new expression graph, so we can call it again with
        # different input data for variable arrays.
        if state['variable_found']:
            args = []
            for i, name in enumerate(state['input_names']):
                if name in state['variable_names']:
                    args.append(VariableDataNode(name=name))
                else:
                    args.append(ArrayDataNode(array_data=state['inputs'][i]))
            self.array_node.data = UFuncNode(ufunc=state['ufunc'], args=args)
            return result

        self.array_node.data = ArrayDataNode(array_data=result)
        return result


    def __str__(self):
        return str(self.eval())

    def __repr__(self):
        return Repr(self.array_node, state={'level':0})

    def __add__(self, other):
        return ufuncs.add(self, other)

    def __radd__(self, other):
        return ufuncs.add(other, self)

    def __iadd__(self, other):
        self.array_node = ufuncs.add(self, other).array_node
        return self

    def __sub__(self, other):
        return ufuncs.subtract(self, other)

    def __rsub__(self, other):
        return ufuncs.subtract(other, self)

    def __mul__(self, other):
        return ufuncs.multiply(self, other)

    def __rmul__(self, other):
        return ufuncs.multiply(other, self)

    def __div__(self, other):
        return ufuncs.divide(self, other)

    def __rdiv__(self, other):
        return ufuncs.divide(other, self)

    def __truediv__(self, other):
        return ufuncs.true_divide(self, other)

    def __rtruediv__(self, other):
        return ufuncs.true_divide(other, self)

    def __floordiv__(self, other):
        return ufuncs.floor_divide(self, other)

    def __rfloordiv__(self, other):
        return ufuncs.floor_divide(other, self)

    def __neg__(self):
        return ufuncs.negative(self)

    def __pow__(self, other):
        return ufuncs.power(self, other)

    def __rpow__(self, other):
        return ufuncs.power(other, self)

    def __getitem__(self, other):
        data = self.eval()
        return Array(data=np.array(data[other]))

    def __setitem__(self, key, value):
        data = self.eval()
        if isinstance(value, Array):
            data[key] = value.eval()
        else:
            data[key] = value
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


from . import ufuncs
