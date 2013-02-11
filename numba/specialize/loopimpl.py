"""
Define various loop implementations.
"""

import ast

import numba
from numba import *
from numba import function_util
from numba import visitors, nodes, error, functions

logger = logging.getLogger(__name__)

iterator_impls = {}

def register_iterator_implementation(iterator_type, iterator_impl):
    iterator_impls[iterator_type] = iterator_impl

class IteratorImpl(object):
    "Implementation of an iterator over a value of a certain type"

    def getiter(self, context, for_node, llvm_module):
        "Set up an iterator (statement or None)"
        raise NotImplementedError

    def body(self, context, for_node, llvm_module):
        "Get the loop body as a list of statements"
        return list(for_node.body)

    def next(self, context, for_node, llvm_module):
        "Get the next iterator element (ExprNode)"
        raise NotImplementedError


class NativeIteratorImpl(IteratorImpl):
    """
    Implement iteration over an iterator which has externally callable
    functions for the `getiter` and `next` operations.
    """

    def __init__(self, getiter_func, next_func):
        self.getiter_func = getiter_func
        self.next_func = next_func
        self.iterator = None

    def getiter(self, context, for_node, llvm_module):
        iterator = function_util.external_call(context, llvm_module,
                                               self.getiter_func,
                                               args=[for_node.iter])
        iterator = nodes.CloneableNode(iterator)
        self.iterator = iterator.clone
        return iterator

    def next(self, context, for_node, llvm_module):
        return function_util.external_call(context, llvm_module,
                                           self.next_func,
                                           args=[self.iterator])


#------------------------------------------------------------------------
# Register Loop Implementations
#------------------------------------------------------------------------

register_iterator_implementation(object_, NativeIteratorImpl("PyObject_GetIter",
                                                             "PyIter_Next"))

