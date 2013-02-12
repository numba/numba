"""
Define various loop implementations.
"""

import ast

import numba
from numba import *
from numba import function_util
from numba import visitors, nodes, error, functions
from numba.typesystem import typematch

logger = logging.getLogger(__name__)

iterator_impls = []

def register_iterator_implementation(iterator_pattern, iterator_impl):
    iterator_impls.append((iterator_pattern, iterator_impl))

def find_iterator_impl(node):
    "Find a suitable iterator type for which we have an implementation"
    type = node.iter.type

    for pattern, impl in iterator_impls:
        if typematch(pattern, type):
            return impl

    raise error.NumbaError(node, "Unsupported iterator "
                                 "type: %s" % (type,))


#------------------------------------------------------------------------
# Interface for Loop Implementations
#------------------------------------------------------------------------

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

