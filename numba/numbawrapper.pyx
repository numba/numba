cimport cython
from numba._numba cimport *
cimport numpy as cnp

import types
import ctypes

import numba
from numba import error
from numba.typesystem import itypesystem
from numba.support import ctypes_support, cffi_support

import numpy as np


cdef extern from *:
    ctypedef struct PyTypeObject:
        PyObject *tp_dict

    ctypedef struct PyMethodDef:
        pass

cdef extern from "numbafunction.h":
    cdef size_t closure_field_offset
    cdef PyTypeObject *NumbaFunctionType
    cdef int NumbaFunction_init() except -1
    cdef object NumbaFunction_NewEx(
            PyMethodDef *ml, module, code, PyObject *closure,
            void *native_func, native_signature, keep_alive)

#------------------------------------------------------------------------
# Numba Function Wrappers
#------------------------------------------------------------------------

cdef class NumbaWrapper(object):
    """
    Numba wrapper function.

        py_func: original Python function
    """

    cdef public object py_func
    cdef public object func_name, func_doc, module

    def __init__(self, py_func):
        self.py_func = py_func

        self.func_name = py_func.__name__
        self.func_doc = py_func.__doc__
        self.module = py_func.__module__

#------------------------------------------------------------------------
# Create Numba Functions (numbafunction.c)
#------------------------------------------------------------------------

NumbaFunction_init()
NumbaFunction_NewEx_pointer = <Py_uintptr_t> &NumbaFunction_NewEx

numbafunction_type = <object> NumbaFunctionType
numbafunc_closure_field_offset = closure_field_offset

def create_function(methoddef, py_func, lfunc_pointer, signature, modname):
    cdef Py_uintptr_t methoddef_p = ctypes.cast(ctypes.byref(methoddef),
                                                ctypes.c_void_p).value
    cdef PyMethodDef *ml = <PyMethodDef *> methoddef_p
    cdef Py_uintptr_t lfunc_p = lfunc_pointer

    result = NumbaFunction_NewEx(ml, modname, getattr(py_func, "__code__", None),
                                 NULL, <void *> lfunc_p, signature, py_func)
    return result

#------------------------------------------------------------------------
# Classes to exclude from type hashing
#------------------------------------------------------------------------

support_classes = (ctypes_support.CData,)
if cffi_support.ffi is not None:
    support_classes += (cffi_support.cffi_func_type,)

cdef tuple hash_on_value_types = (
    itypesystem.Type,
    np.ufunc,
    np.dtype,
    NumbaWrapper,
    types.FunctionType,
    types.BuiltinFunctionType,
    types.MethodType,
    getattr(types, 'UnboundMethodType', types.FunctionType),
    types.BuiltinMethodType,
) # + support_classes

def add_hash_by_value_type(type):
    global hash_on_value_types

    hash_on_value_types += (type,)

#------------------------------------------------------------------------
# @jit function creation
#------------------------------------------------------------------------

cdef class NumbaCompiledWrapper(NumbaWrapper):
    """
    Temporary numba wrapper function for @jit, only used for recursion.

        signature: minitype function signature
        lfunc: LLVM function
    """

    cdef public object lfunc, signature, wrapper, lfunc_pointer

    def __init__(self, py_func, signature, lfunc):
        super(NumbaCompiledWrapper, self).__init__(py_func)

        self.signature = signature
        self.lfunc = lfunc
        self.lfunc_pointer = None

    def __repr__(self):
        return '<compiled numba function (%s) :: %s>' % (self.py_func,
                                                         self.signature)

def create_numba_wrapper(py_func, numbafunction, signature, lfunc):
    """
    Use the NumbaFunction to set attributes for numba, and return the
    NumbaFunction.
    """
    if numbafunction is None:
        # Function is called recursively, use a placeholder
        return NumbaCompiledWrapper(py_func, signature, lfunc)

    # lfunc_pointer is set by create_function()
    numbafunction.py_func = py_func
    numbafunction.signature = signature
    numbafunction.lfunc = lfunc
    return numbafunction

def is_numba_wrapper(numbafunction):
    """
    Check whether the given object is a numba function wrapper around a
    compiled function.
    """
    return isinstance(numbafunction, (NumbaCompiledWrapper, numbafunction_type))

#------------------------------------------------------------------------
# Numba Autojit Function Wrapper
#------------------------------------------------------------------------

cdef class _NumbaSpecializingWrapper(NumbaWrapper):
    """
    Numba wrapper function for @autojit. Specializes py_func when called with
    new argtypes.

        py_func: original Python function
        compiler: numba.wrapper.compiler.Compiler
                  Compiles the function given the argument values.
        funccache: AutojitFunctionCache that can quickly lookup the right
                   specialization
    """

    cdef public AutojitFunctionCache funccache
    cdef public object compiler

    def __init__(self, py_func, compiler, funccache):
        super(_NumbaSpecializingWrapper, self).__init__(py_func)
        self.compiler = compiler
        self.funccache = funccache

    def __repr__(self):
        return '<specializing numba function(%s)>' % self.py_func

    def add_specialization(self, signature):
        numba_wrapper = self.compiler.compile(signature)

        # NOTE: We do not always have args (e.g. if one autojit function
        # or method calls another one). However, the first call from Python
        # will retrieve it from the slow cache (env.specializations)

        # self.funccache.add(args, numba_wrapper)

        return numba_wrapper

    def __call__(self, *args, **kwargs):
        if len(kwargs):
            raise error.NumbaError("Cannot handle keyword arguments yet")

        numba_wrapper = self.funccache.lookup(args)
        if numba_wrapper is None:
            # print "Cache miss for function:", self.py_func.__name__
            numba_wrapper = self.compiler.compile_from_args(args, kwargs)
            self.funccache.add(args, numba_wrapper)

        return PyObject_Call(<PyObject *> numba_wrapper,
                             <PyObject *> args, NULL)

    def __get__(self, instance, type):
        if instance is None:
            if numba.PY3:
                return UnboundFunctionWrapper(self, type)
            else:
                return self

        return BoundSpecializingWrapper(self, instance)


class NumbaSpecializingWrapper(_NumbaSpecializingWrapper):
    # Don't make this a docstring, it breaks the __doc__ propertyr
    # """
    # Python class to allow overriding properties such as __name__.
    # """

    @property
    def __name__(self):
        return self.func_name

    @property
    def __doc__(self):
        return self.func_doc

    @property
    def __module__(self):
        return self.module


#------------------------------------------------------------------------
# Unbound Methods
#------------------------------------------------------------------------

def unbound_method_type_check(py_class, obj):
    if not isinstance(obj, py_class):
        raise TypeError(
            "unbound method numba_function_or_method object must be "
            "called with %s instance as first argument "
            "(got %s instance instead)" % (py_class.__name__, type(obj).__name__))


cdef class UnboundFunctionWrapper(object):
    """
    Wraps unbound functions in Python 3, for jit and autojit methods.
    PyInstanceMethod does not check whether 'self' is an instance of 'type'.

    Hence the following works in Python 3:

        class C(object):
            def m(self):
                print self

        C.m(object())

    However, this is dangerous for numba code, since it expects an instance
    of 'C' (and may access fields in a low-level way).
    """

    cdef public object func, type

    def __init__(self, func, type):
        self.func = func
        self.type = type

    def __call__(self, *args, **kwargs):
        assert len(args) > 0, "Unbound method must have at least one argument"
        unbound_method_type_check(self.type, args[0])
        return self.func(*args, **kwargs)


cdef public Create_NumbaUnboundMethod(PyObject *func, PyObject *type):
    assert type != NULL
    obj = UnboundFunctionWrapper(<object> func, <object> type)
    return obj

#------------------------------------------------------------------------
# Bound Methods
#------------------------------------------------------------------------

cdef class BoundSpecializingWrapper(object):
    """
    Numba wrapper created for bound @autojit methods. Note that @jit methods
    don't need this, since numbafunction does the binding.
    """

    cdef public object specializing_wrapper, type, instance

    def __init__(self, specializing_wrapper, instance):
        self.specializing_wrapper = specializing_wrapper
        self.instance = instance

    def __call__(self, *args, **kwargs):
        return self.specializing_wrapper(self.instance, *args, **kwargs)

#------------------------------------------------------------------------
# Autojit Fast Function Cache
#------------------------------------------------------------------------

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

    cdef public dict specializations

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
