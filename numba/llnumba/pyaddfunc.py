#! /usr/bin/env python
# ______________________________________________________________________

import ctypes

# ______________________________________________________________________
class PyMethodDef (ctypes.Structure):
    _fields_ = [
        ('ml_name', ctypes.c_char_p),
        ('ml_meth', ctypes.c_void_p),
        ('ml_flags', ctypes.c_int),
        ('ml_doc', ctypes.c_char_p),
        ]

PyCFunction_NewEx = ctypes.pythonapi.PyCFunction_NewEx
PyCFunction_NewEx.argtypes = (ctypes.POINTER(PyMethodDef),
                              ctypes.c_void_p,
                              ctypes.c_void_p)
PyCFunction_NewEx.restype = ctypes.py_object

cache = {} # Unsure if this is necessary to keep the PyMethodDef
           # structures from being garbage collected.  Assuming so...

def pyaddfunc (func_name, func_ptr, func_doc = None):
    global cache
    if bytes != str:
        func_name = bytes(ord(ch) for ch in func_name)
    key = (func_name, func_ptr)
    if key in cache:
        _, ret_val = cache[key]
    else:
        mdef = PyMethodDef(bytes(func_name),
                           func_ptr,
                           1, # == METH_VARARGS (hopefully remains so...)
                           func_doc)
        ret_val = PyCFunction_NewEx(ctypes.byref(mdef), 0, 0)
        cache[key] = (mdef, ret_val)
    return ret_val

# ______________________________________________________________________
# End of pyaddfunc.py
