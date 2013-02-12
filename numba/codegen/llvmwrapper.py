"""
Module that creates wrapper around llvm functions. The wrapper is callable
from Python.
"""

import ctypes

import llvm.core

from numba import *
from numba import extension_types
from numba.functions import keep_alive

def _create_methoddef(py_func, func_name, func_doc, func_pointer):
    # struct PyMethodDef {
    #     const char  *ml_name;   /* The name of the built-in function/method */
    #     PyCFunction  ml_meth;   /* The C function that implements it */
    #     int      ml_flags;      /* Combination of METH_xxx flags, which mostly
    #                                describe the args expected by the C func */
    #     const char  *ml_doc;    /* The __doc__ attribute, or NULL */
    # };
    PyMethodDef = struct([('name', c_string_type),
                          ('method', void.pointer()),
                          ('flags', int_),
                          ('doc', c_string_type)])
    c_PyMethodDef = PyMethodDef.to_ctypes()

    PyCFunction_NewEx = ctypes.pythonapi.PyCFunction_NewEx
    PyCFunction_NewEx.argtypes = [ctypes.POINTER(c_PyMethodDef),
                                  ctypes.py_object,
                                  ctypes.c_void_p]
    PyCFunction_NewEx.restype = ctypes.py_object

    # It is paramount to put these into variables first, since every
    # access may return a new string object!
    keep_alive(py_func, func_name)
    keep_alive(py_func, func_doc)

    methoddef = c_PyMethodDef()
    if PY3:
        if func_name is not None:
            func_name = func_name.encode('utf-8')
        if func_doc is not None:
            func_doc = func_doc.encode('utf-8')

    methoddef.name = func_name
    methoddef.doc = func_doc
    methoddef.method = ctypes.c_void_p(func_pointer)
    methoddef.flags = 1 # METH_VARARGS

    return methoddef

def numbafunction_new(py_func, func_name, func_doc, module_name, func_pointer,
                      wrapped_lfunc_pointer, wrapped_signature):
    methoddef = _create_methoddef(py_func, func_name, func_doc, func_pointer)
    keep_alive(py_func, methoddef)
    keep_alive(py_func, module_name)

    wrapper = extension_types.create_function(methoddef, py_func,
                                              wrapped_lfunc_pointer,
                                              wrapped_signature, module_name)
    return methoddef, wrapper
