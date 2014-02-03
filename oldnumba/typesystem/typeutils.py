# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import

from numba import error
from numba.typesystem import *

#------------------------------------------------------------------------
# Utilities
#------------------------------------------------------------------------

ts = numba_typesystem

def is_obj(type):
    return type.is_object or type.is_array

native_type_dict = {}
for native_type in native_integral:
    if native_type not in (Py_ssize_t, npy_intp, Py_uintptr_t, size_t): # TODO: do this better
        native_type_dict[(native_type.itemsize, native_type.signed)] = native_type

def promote_to_native(int_type):
    return native_type_dict[int_type.itemsize, int_type.signed]

def promote_closest(ts, int_type, candidates):
    """
    promote_closest(Py_ssize_t, [int_, long_, longlong]) -> longlong
    """
    for candidate in candidates:
        promoted = ts.promote(int_type, candidate)
        if promoted.itemsize == candidate.itemsize and promoted.signed == candidate.signed:
            return candidate

    return candidates[-1]

def get_type(ast_node):
    """
    :param ast_node: a Numba or Python AST expression node
    :return: the type of the expression node
    """
    return ast_node.variable.type


def error_index(type):
    raise error.NumbaError("Type %s can not be indexed or "
                           "iterated over" % (type,))


def index_type(type):
    "Result of indexing a value of the given type with an integer index"
    if type.is_array:
        result = array(type.dtype, type.ndim - 1)
    elif type.is_container or type.is_pointer or type.is_carray:
        result = type.base_type
    elif type.is_dict:
        result = type.value_type
    elif type.is_range:
        result = Py_ssize_t
    elif type.is_object:
        result = object_
    else:
        error_index(type)

    return result

def element_type(type):
    "Result type of iterating over something"
    if type.is_dict:
        return type.key_type
    elif type.is_pointer and not type.is_sized_pointer:
        error_index(type)
    else:
        return index_type(type)

def require(ast_nodes, properties):
    "Assert that the types of the given nodes meets a certain requirement"
    for ast_node in ast_nodes:
        if not any(getattr(get_type(ast_node), p) for p in properties):
            typenames = ", or ".join(p[3:] for p in properties) # remove 'is_' prefix
            raise error.NumbaError(ast_node, "Expected an %s" % (typenames,))

def pyfunc_signature(nargs):
    "Signature of a python function with N arguments"
    return function(args=(object_,) * nargs, return_type=object_)
