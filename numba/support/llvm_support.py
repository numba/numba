# -*- coding: utf-8 -*-

"""
Bind LLVM functions to ctypes.
Originally adapated from Bitey, blaze/blir/bind.py.
"""

from __future__ import print_function, division, absolute_import

from llvm.core import Module
import llvm.core
import llvm.ee
import ctypes
import io
import os
import sys

PY3 = sys.version_info[0] >= 3

def map_llvm_to_ctypes(llvm_type, py_module=None):
    '''
    Map an LLVM type to an equivalent ctypes type. py_module is an
    optional module that is used for structure wrapping.  If
    structures are found, the struct definitions will be created in
    that module.
    '''
    kind = llvm_type.kind
    if kind == llvm.core.TYPE_INTEGER:
        ctype = getattr(ctypes,"c_int"+str(llvm_type.width))

    elif kind == llvm.core.TYPE_DOUBLE:
        ctype = ctypes.c_double

    elif kind == llvm.core.TYPE_FLOAT:
        ctype = ctypes.c_float

    elif kind == llvm.core.TYPE_VOID:
        ctype = None

    elif kind == llvm.core.TYPE_POINTER:
        pointee = llvm_type.pointee
        p_kind = pointee.kind
        if p_kind == llvm.core.TYPE_INTEGER:
            width = pointee.width

            # Special case:  char * is mapped to strings
            if width == 8:
                ctype = ctypes.c_char_p
            else:
                ctype = ctypes.POINTER(map_llvm_to_ctypes(pointee, py_module))

        # Special case: void * mapped to c_void_p type
        elif p_kind == llvm.core.TYPE_VOID:
            ctype = ctypes.c_void_p
        else:
            ctype = ctypes.POINTER(map_llvm_to_ctypes(pointee, py_module))

    elif kind == llvm.core.TYPE_STRUCT:
        struct_name = llvm_type.name.split('.')[-1]
        if not PY3:
            struct_name = struct_name.encode('ascii')

        # If the named type is already known, return it
        if py_module:
            struct_type = getattr(py_module, struct_name, None)
        else:
            struct_type = None

        if struct_type and issubclass(struct_type, ctypes.Structure):
            return struct_type

        # If there is an object with the name of the structure already present and it has
        # the field names specified, use those names to help out
        if hasattr(struct_type, '_fields_'):
            names = struct_type._fields_
        else:
            names = [ "e"+str(n) for n in range(llvm_type.element_count) ]

        # Create a class definition for the type. It is critical that this
        # Take place before the handling of members to avoid issues with
        # self-referential data structures
        ctype = type(ctypes.Structure)(struct_name, (ctypes.Structure,),
                                       { '__module__' : py_module.__name__ })

        if py_module:
            setattr(py_module, struct_name, ctype)

        # Resolve the structure fields
        fields = [ (name, map_llvm_to_ctypes(elem, py_module))
                   for name, elem in zip(names, llvm_type.elements) ]

        # Set the fields member of the type last.  The order is critical
        # to deal with self-referential structures.
        setattr(ctype, '_fields_', fields)
    else:
        raise TypeError("Unknown type %s" % llvm_type)

    return ctype

def wrap_llvm_function(func, engine, py_module):
    '''
    Create a ctypes wrapper around an LLVM function.
    engine is the LLVM execution engine.
    func is an LLVM function instance.
    py_module is a Python module where to put the wrappers.
    '''
    args = func.type.pointee.args
    ret_type = func.type.pointee.return_type
    try:
        ret_ctype = map_llvm_to_ctypes(ret_type, py_module)
        args_ctypes = [map_llvm_to_ctypes(arg, py_module) for arg in args]
    except TypeError as e:
        if 'BITEYDEBUG' in os.environ:
            print("Couldn't wrap %s. Reason %s" % (func.name, e))
        return None

    # Declare the ctypes function prototype
    functype = ctypes.CFUNCTYPE(ret_ctype, *args_ctypes)

    # Get the function point from the execution engine
    addr = engine.get_pointer_to_function(func)

    # Make a ctypes callable out of it
    wrapper = functype(addr)

    # Set it in the module
    setattr(py_module, func.name, wrapper)
    wrapper.__name__ = func.name

def wrap_llvm_module(llvm_module, engine, py_module):
    '''
    Build ctypes wrappers around an existing LLVM module and execution
    engine.  py_module is an existing Python module that will be
    populated with the resulting wrappers.
    '''
    functions = [func for func in llvm_module.functions
                 if not func.name.startswith("_")
                 and not func.is_declaration
                 and func.linkage == llvm.core.LINKAGE_EXTERNAL]
    for func in functions:
        wrap_llvm_function(func, engine, py_module)

def wrap_llvm_bitcode(bitcode, py_module):
    '''
    Given a byte-string of LLVM bitcode and a Python module,
    populate the module with ctypes bindings for public methods
    in the bitcode.
    '''
    llvm_module = Module.from_bitcode(io.BytesIO(bitcode))
    engine = llvm.ee.ExecutionEngine.new(llvm_module)
    wrap_llvm_module(llvm_module, engine, py_module)
    setattr(py_module, '_llvm_module', llvm_module)
    setattr(py_module, '_llvm_engine', engine)
