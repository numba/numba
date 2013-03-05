cimport cython
from numba._numba cimport *
cimport numpy as cnp

import types
import ctypes

from numba import error
from numba.minivect import minitypes
from numba.support import ctypes_support, cffi_support

import numpy as np

#------------------------------------------------------------------------
# Create Numba Functions (numbafunction.c)
#------------------------------------------------------------------------

cdef extern from *:
    ctypedef struct PyMethodDef:
        pass

cdef extern from "numbafunction.h":
    cdef size_t closure_field_offset
    cdef int NumbaFunction_init() except -1
    cdef object NumbaFunction_NewEx(
            PyMethodDef *ml, module, code, PyObject *closure,
            void *native_func, native_signature, keep_alive)

NumbaFunction_init()
NumbaFunction_NewEx_pointer = <Py_uintptr_t> &NumbaFunction_NewEx

numbafunc_closure_field_offset = closure_field_offset

def create_function(methoddef, py_func, lfunc_pointer, signature, modname):
    cdef Py_uintptr_t methoddef_p = ctypes.cast(ctypes.byref(methoddef),
                                                ctypes.c_void_p).value
    cdef PyMethodDef *ml = <PyMethodDef *> methoddef_p
    cdef Py_uintptr_t lfunc_p = lfunc_pointer

    result = NumbaFunction_NewEx(ml, modname, getattr(py_func, "func_code", None),
                                 NULL, <void *> lfunc_p, signature, py_func)
    return result


#------------------------------------------------------------------------
# Classes to exclude from type hashing
#------------------------------------------------------------------------

support_classes = (ctypes_support.CData,)
if cffi_support.ffi is not None:
    support_classes += (cffi_support.cffi_func_type,)

cdef tuple hash_on_value_types = (
    minitypes.Type,
    np.ufunc,
    np.dtype,
    np.generic,
    NumbaWrapper,
    types.FunctionType,
    types.BuiltinFunctionType,
    types.MethodType,
    getattr(types, 'UnboundMethodType', types.FunctionType),
    types.BuiltinMethodType,
) # + support_classes

#------------------------------------------------------------------------
# Create Numba Functions (numbafunction.c)
#------------------------------------------------------------------------


cdef class NumbaWrapper(object):
    """
    Numba wrapper function.

        py_func: original Python function
    """
    cdef public object py_func
    cdef public object func_name, func_doc
    cdef public object __name__, __doc__, __module__

    def __init__(self, py_func):
        self.py_func = py_func

        self.func_name = self.__name__ = py_func.__name__
        self.func_doc = self.__doc__ = py_func.__doc__
        self.__module__ = py_func.__module__


cdef class NumbaCompiledWrapper(NumbaWrapper):
    """
    Numba wrapper function for @jit.

        wrapper: LLVM function wrapper, callable from Python
        signature: minitype FunctionType signature
        lfunc: LLVM function
    """

    cdef public object lfunc, signature, wrapper

    def __init__(self, py_func, wrapper, signature, lfunc):
        super(NumbaCompiledWrapper, self).__init__(py_func)

        self.wrapper = wrapper
        self.signature = signature
        self.lfunc = lfunc

    def __repr__(self):
        return '<compiled numba function (%s) :: %s>' % (self.py_func,
                                                         self.signature)

    def __call__(self, *args, **kwargs):
        if len(kwargs):
            raise error.NumbaError("Cannot handle keyword arguments yet")

        return PyObject_Call(<PyObject *> self.wrapper,
                             <PyObject *> args, NULL)

cdef class NumbaSpecializingWrapper(NumbaWrapper):
    """
    Numba wrapper function for @autojit.

        py_func: original Python function
        compiling_decorator: function that can compile the function given
                             the argument values.
                             (args, kwargs) -> compiled_callable
        funccache: AutojitFunctionCache that can quickly lookup the right
                   specialization
    """

    cdef public AutojitFunctionCache funccache
    cdef public object compiling_decorator

    def __init__(self, py_func, compiling_decorator, funccache):
        super(NumbaSpecializingWrapper, self).__init__(py_func)
        self.compiling_decorator = compiling_decorator
        self.funccache = funccache

    def __repr__(self):
        return '<specializing numba function(%s)>' % self.py_func

    def __call__(self, *args, **kwargs):
        cdef NumbaCompiledWrapper numba_wrapper

        if len(kwargs):
            raise error.NumbaError("Cannot handle keyword arguments yet")

        numba_wrapper = <NumbaCompiledWrapper> self.funccache.lookup(args)
        if numba_wrapper is None:
            # print "Cache miss for function:", self.py_func.__name__
            numba_wrapper = self.compiling_decorator(args, kwargs)
            self.funccache.add(args, numba_wrapper)

        return PyObject_Call(<PyObject *> numba_wrapper.wrapper,
                             <PyObject *> args, NULL)


cdef inline _id(obj):
    return <Py_uintptr_t> <PyObject *> obj

cdef inline void setkey(t, int i, k):
    Py_INCREF(<PyObject *> k)
    PyTuple_SET_ITEM(t, i, k)

cpdef inline getkey(tuple args): # 3.0x
    """
    Get the tuple key we need to look up the right specialization from the
    runtime autojit arguments.

    We micro-optimize this to avoid significant overhead for short functions
    (the dispatch may in fact be significantly more expensive than the actual
     function call).
    """
    cdef Py_ssize_t i

    cdef Py_ssize_t nargs = PyTuple_GET_SIZE(args)
    cdef tuple key = PyTuple_New(nargs * 3)
    cdef cnp.ndarray array

    for i in range(nargs):
        arg = <object> PyTuple_GET_ITEM(args, i)
        # arg = args[i]
        if isinstance(arg, cnp.ndarray):
            # Also specialize of C/F data when we implement optimizations
            # for those.
            array = <cnp.ndarray> arg

            # Hashing on dtype here makes the test example more than
            # twice as slow, hash on its id() instead.
            # k = (type(arg), array.descr, array.ndim)
            # k = (type(arg), _id(array.descr), array.ndim)
            setkey(key, i*3, type(arg))
            setkey(key, i*3+1, _id(array.descr))
            # NumPy maximum ndim is 32 (2 ** 5).
            setkey(key, i*3+2, array.ndim | (cnp.PyArray_FLAGS(arg) << 5))
            continue

        elif isinstance(arg, hash_on_value_types):
            # A type is passed in as a value, hash the thing (this will be slow)
            setkey(key, i*3, arg)

        else:
            # k = type(arg)
            setkey(key, i*3, type(arg))

        setkey(key, i*3+1, 0)
        setkey(key, i*3+2, 0)

        # Py_INCREF(<PyObject *> k)
        # PyTuple_SET_ITEM(key, i, k)

    return key


cdef class AutojitFunctionCache(object):
    """
    Try a faster lookup for autojit functions.

    This function cache may give none where a compiled specialization does
    exist. This is caught by the slow path going through
    functions.FunctionCache.
    """

    cdef dict specializations

    # list of dtypes that need to be alive in order for the id() hash to
    # remain valid
    cdef list dtypes

    def __init__(self):
        self.specializations = {}
        self.dtypes = []

    cpdef add(self, args, wrapper):
        # self.specializations[0] = wrapper
#        key = (0x19228, 0x384726)
        key = getkey(args)
        self.specializations[key] = wrapper

        for arg in args:
            if isinstance(arg, np.ndarray):
                self.dtypes.append(arg.dtype)

    cdef lookup(self, tuple args):
        # return self.specializations[0]
        key = getkey(args)

#        key = (0x19228, 0x384726)
        wrapper = self.specializations.get(key)
        return wrapper
