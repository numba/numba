from __future__ import print_function, division, absolute_import

import llvmlite.binding as ll
from llvmlite.llvmpy.core import Type, Constant, LLVMException
import llvmlite.llvmpy.core as lc

from numba.config import PYVERSION
import numba.ctypes_support as ctypes
from numba import types, utils, cgutils, _helperlib, assume

_PyNone = ctypes.c_ssize_t(id(None))


class NativeError(RuntimeError):
    pass


@utils.runonce
def fix_python_api():
    """
    Execute once to install special symbols into the LLVM symbol table
    """

    ll.add_symbol("Py_None", ctypes.addressof(_PyNone))
    ll.add_symbol("numba_native_error", id(NativeError))

    # Add C helper functions
    c_helpers = _helperlib.c_helpers
    for py_name in c_helpers:
        c_name = "numba_" + py_name
        c_address = c_helpers[py_name]
        ll.add_symbol(c_name, c_address)

    # Add all built-in exception classes
    for obj in utils.builtins.__dict__.values():
        if isinstance(obj, type) and issubclass(obj, BaseException):
            ll.add_symbol("PyExc_%s" % (obj.__name__), id(obj))


class PythonAPI(object):
    """
    Code generation facilities to call into the CPython C API (and related
    helpers).
    """

    def __init__(self, context, builder):
        """
        Note: Maybe called multiple times when lowering a function
        """
        fix_python_api()
        self.context = context
        self.builder = builder

        self.module = builder.basic_block.function.module
        # Initialize types
        self.pyobj = self.context.get_argument_type(types.pyobject)
        self.voidptr = Type.pointer(Type.int(8))
        self.long = Type.int(ctypes.sizeof(ctypes.c_long) * 8)
        self.ulonglong = Type.int(ctypes.sizeof(ctypes.c_ulonglong) * 8)
        self.longlong = self.ulonglong
        self.double = Type.double()
        self.py_ssize_t = self.context.get_value_type(types.intp)
        self.cstring = Type.pointer(Type.int(8))
        self.gil_state = Type.int(_helperlib.py_gil_state_size * 8)

    # ------ Python API -----

    #
    # Basic object API
    #

    def incref(self, obj):
        fnty = Type.function(Type.void(), [self.pyobj])
        fn = self._get_function(fnty, name="Py_IncRef")
        self.builder.call(fn, [obj])

    def decref(self, obj):
        fnty = Type.function(Type.void(), [self.pyobj])
        fn = self._get_function(fnty, name="Py_DecRef")
        self.builder.call(fn, [obj])

    #
    # Argument unpacking
    #

    def parse_tuple_and_keywords(self, args, kws, fmt, keywords, *objs):
        charptr = Type.pointer(Type.int(8))
        charptrary = Type.pointer(charptr)
        argtypes = [self.pyobj, self.pyobj, charptr, charptrary]
        fnty = Type.function(Type.int(), argtypes, var_arg=True)
        fn = self._get_function(fnty, name="PyArg_ParseTupleAndKeywords")
        return self.builder.call(fn, [args, kws, fmt, keywords] + list(objs))

    def parse_tuple(self, args, fmt, *objs):
        charptr = Type.pointer(Type.int(8))
        argtypes = [self.pyobj, charptr]
        fnty = Type.function(Type.int(), argtypes, var_arg=True)
        fn = self._get_function(fnty, name="PyArg_ParseTuple")
        return self.builder.call(fn, [args, fmt] + list(objs))

    #
    # Exception handling
    #

    def err_occurred(self):
        fnty = Type.function(self.pyobj, ())
        fn = self._get_function(fnty, name="PyErr_Occurred")
        return self.builder.call(fn, ())

    def err_clear(self):
        fnty = Type.function(Type.void(), ())
        fn = self._get_function(fnty, name="PyErr_Clear")
        return self.builder.call(fn, ())

    def err_set_string(self, exctype, msg):
        fnty = Type.function(Type.void(), [self.pyobj, self.cstring])
        fn = self._get_function(fnty, name="PyErr_SetString")
        if isinstance(exctype, str):
            exctype = self.get_c_object(exctype)
        if isinstance(msg, str):
            msg = self.context.insert_const_string(self.module, msg)
        return self.builder.call(fn, (exctype, msg))

    def err_set_object(self, exctype, excval):
        fnty = Type.function(Type.void(), [self.pyobj, self.pyobj])
        fn = self._get_function(fnty, name="PyErr_SetObject")
        return self.builder.call(fn, (exctype, excval))

    def err_write_unraisable(self, obj):
        fnty = Type.function(Type.void(), [self.pyobj])
        fn = self._get_function(fnty, name="PyErr_WriteUnraisable")
        return self.builder.call(fn, (obj,))

    def raise_native_error(self, msg):
        cstr = self.context.insert_const_string(self.module, msg)
        self.err_set_string(self.native_error_type, cstr)

    def raise_exception(self, exctype, excval):
        # XXX This produces non-reusable bitcode: the pointer's value
        # is specific to this process execution.
        exctypeaddr = self.context.get_constant(types.intp, id(exctype))
        excvaladdr = self.context.get_constant(types.intp, id(excval))
        self.err_set_object(exctypeaddr.inttoptr(self.pyobj),
                            excvaladdr.inttoptr(self.pyobj))

    def get_c_object(self, name):
        """
        Get a Python object through its C-accessible *name*.
        (e.g. "PyExc_ValueError").
        """
        try:
            gv = self.module.get_global_variable_named(name)
        except LLVMException:
            gv = self.module.add_global_variable(self.pyobj.pointee, name)
        return gv

    @property
    def native_error_type(self):
        return self.get_c_object("numba_native_error")

    def raise_missing_global_error(self, name):
        msg = "global name '%s' is not defined" % name
        cstr = self.context.insert_const_string(self.module, msg)
        self.err_set_string("PyExc_NameError", cstr)

    def raise_missing_name_error(self, name):
        msg = "name '%s' is not defined" % name
        cstr = self.context.insert_const_string(self.module, msg)
        self.err_set_string("PyExc_NameError", cstr)

    #
    # Concrete dict API
    #

    def dict_getitem_string(self, dic, name):
        """Lookup name inside dict

        Returns a borrowed reference
        """
        fnty = Type.function(self.pyobj, [self.pyobj, self.cstring])
        fn = self._get_function(fnty, name="PyDict_GetItemString")
        cstr = self.context.insert_const_string(self.module, name)
        return self.builder.call(fn, [dic, cstr])

    def dict_getitem(self, dic, name):
        """Lookup name inside dict

        Returns a borrowed reference
        """
        fnty = Type.function(self.pyobj, [self.pyobj, self.pyobj])
        fn = self._get_function(fnty, name="PyDict_GetItem")
        return self.builder.call(fn, [dic, name])

    def dict_new(self, presize=0):
        if presize == 0:
            fnty = Type.function(self.pyobj, ())
            fn = self._get_function(fnty, name="PyDict_New")
            return self.builder.call(fn, ())
        else:
            fnty = Type.function(self.pyobj, [self.py_ssize_t])
            fn = self._get_function(fnty, name="_PyDict_NewPresized")
            return self.builder.call(fn,
                                     [Constant.int(self.py_ssize_t, presize)])

    def dict_setitem(self, dictobj, nameobj, valobj):
        fnty = Type.function(Type.int(), (self.pyobj, self.pyobj,
                                          self.pyobj))
        fn = self._get_function(fnty, name="PyDict_SetItem")
        return self.builder.call(fn, (dictobj, nameobj, valobj))

    def dict_setitem_string(self, dictobj, name, valobj):
        fnty = Type.function(Type.int(), (self.pyobj, self.cstring,
                                          self.pyobj))
        fn = self._get_function(fnty, name="PyDict_SetItemString")
        cstr = self.context.insert_const_string(self.module, name)
        return self.builder.call(fn, (dictobj, cstr, valobj))

    def dict_pack(self, keyvalues):
        """
        Args
        -----
        keyvalues: iterable of (str, llvm.Value of PyObject*)
        """
        dictobj = self.dict_new()
        not_null = cgutils.is_not_null(self.builder, dictobj)
        with cgutils.if_likely(self.builder, not_null):
            for k, v in keyvalues:
                self.dict_setitem_string(dictobj, k, v)
        return dictobj

    #
    # Concrete number APIs
    #

    def float_from_double(self, fval):
        fnty = Type.function(self.pyobj, [self.double])
        fn = self._get_function(fnty, name="PyFloat_FromDouble")
        return self.builder.call(fn, [fval])

    def number_as_ssize_t(self, numobj):
        fnty = Type.function(self.py_ssize_t, [self.pyobj])
        fn = self._get_function(fnty, name="PyNumber_AsSsize_t")
        return self.builder.call(fn, [numobj])

    def number_long(self, numobj):
        fnty = Type.function(self.pyobj, [self.pyobj])
        fn = self._get_function(fnty, name="PyNumber_Long")
        return self.builder.call(fn, [numobj])

    def long_as_ulonglong(self, numobj):
        fnty = Type.function(self.ulonglong, [self.pyobj])
        fn = self._get_function(fnty, name="PyLong_AsUnsignedLongLong")
        return self.builder.call(fn, [numobj])

    def long_as_longlong(self, numobj):
        fnty = Type.function(self.ulonglong, [self.pyobj])
        fn = self._get_function(fnty, name="PyLong_AsLongLong")
        return self.builder.call(fn, [numobj])

    def _long_from_native_int(self, ival, func_name, native_int_type,
                              signed):
        fnty = Type.function(self.pyobj, [native_int_type])
        fn = self._get_function(fnty, name=func_name)
        resptr = cgutils.alloca_once(self.builder, self.pyobj)

        if PYVERSION < (3, 0):
            # Under Python 2, we try to return a PyInt object whenever
            # the given number fits in a C long.
            pyint_fnty = Type.function(self.pyobj, [self.long])
            pyint_fn = self._get_function(pyint_fnty, name="PyInt_FromLong")
            long_max = Constant.int(native_int_type, _helperlib.long_max)
            if signed:
                long_min = Constant.int(native_int_type, _helperlib.long_min)
                use_pyint = self.builder.and_(
                    self.builder.icmp(lc.ICMP_SGE, ival, long_min),
                    self.builder.icmp(lc.ICMP_SLE, ival, long_max),
                    )
            else:
                use_pyint = self.builder.icmp(lc.ICMP_ULE, ival, long_max)

            with cgutils.ifelse(self.builder, use_pyint) as (then, otherwise):
                with then:
                    downcast_ival = self.builder.trunc(ival, self.long)
                    res = self.builder.call(pyint_fn, [downcast_ival])
                    self.builder.store(res, resptr)
                with otherwise:
                    res = self.builder.call(fn, [ival])
                    self.builder.store(res, resptr)
        else:
            fn = self._get_function(fnty, name=func_name)
            self.builder.store(self.builder.call(fn, [ival]), resptr)

        return self.builder.load(resptr)

    def long_from_long(self, ival):
        if PYVERSION < (3, 0):
            func_name = "PyInt_FromLong"
        else:
            func_name = "PyLong_FromLong"
        fnty = Type.function(self.pyobj, [self.long])
        fn = self._get_function(fnty, name=func_name)
        return self.builder.call(fn, [ival])

    def long_from_ssize_t(self, ival):
        return self._long_from_native_int(ival, "PyLong_FromSsize_t",
                                          self.py_ssize_t, signed=True)

    def long_from_longlong(self, ival):
        return self._long_from_native_int(ival, "PyLong_FromLongLong",
                                          self.longlong, signed=True)

    def long_from_ulonglong(self, ival):
        return self._long_from_native_int(ival, "PyLong_FromUnsignedLongLong",
                                          self.ulonglong, signed=False)

    def _get_number_operator(self, name):
        fnty = Type.function(self.pyobj, [self.pyobj, self.pyobj])
        fn = self._get_function(fnty, name="PyNumber_%s" % name)
        return fn

    def _call_number_operator(self, name, lhs, rhs, inplace=False):
        if inplace:
            name = "InPlace" + name
        fn = self._get_number_operator(name)
        return self.builder.call(fn, [lhs, rhs])

    def number_add(self, lhs, rhs, inplace=False):
        return self._call_number_operator("Add", lhs, rhs, inplace=inplace)

    def number_subtract(self, lhs, rhs, inplace=False):
        return self._call_number_operator("Subtract", lhs, rhs, inplace=inplace)

    def number_multiply(self, lhs, rhs, inplace=False):
        return self._call_number_operator("Multiply", lhs, rhs, inplace=inplace)

    def number_divide(self, lhs, rhs, inplace=False):
        assert PYVERSION < (3, 0)
        return self._call_number_operator("Divide", lhs, rhs, inplace=inplace)

    def number_truedivide(self, lhs, rhs, inplace=False):
        return self._call_number_operator("TrueDivide", lhs, rhs, inplace=inplace)

    def number_floordivide(self, lhs, rhs, inplace=False):
        return self._call_number_operator("FloorDivide", lhs, rhs, inplace=inplace)

    def number_remainder(self, lhs, rhs, inplace=False):
        return self._call_number_operator("Remainder", lhs, rhs, inplace=inplace)

    def number_lshift(self, lhs, rhs, inplace=False):
        return self._call_number_operator("Lshift", lhs, rhs, inplace=inplace)

    def number_rshift(self, lhs, rhs, inplace=False):
        return self._call_number_operator("Rshift", lhs, rhs, inplace=inplace)

    def number_and(self, lhs, rhs, inplace=False):
        return self._call_number_operator("And", lhs, rhs, inplace=inplace)

    def number_or(self, lhs, rhs, inplace=False):
        return self._call_number_operator("Or", lhs, rhs, inplace=inplace)

    def number_xor(self, lhs, rhs, inplace=False):
        return self._call_number_operator("Xor", lhs, rhs, inplace=inplace)

    def number_power(self, lhs, rhs, inplace=False):
        fnty = Type.function(self.pyobj, [self.pyobj] * 3)
        fname = "PyNumber_InPlacePower" if inplace else "PyNumber_Power"
        fn = self._get_function(fnty, fname)
        return self.builder.call(fn, [lhs, rhs, self.borrow_none()])

    def number_negative(self, obj):
        fnty = Type.function(self.pyobj, [self.pyobj])
        fn = self._get_function(fnty, name="PyNumber_Negative")
        return self.builder.call(fn, (obj,))

    def number_positive(self, obj):
        fnty = Type.function(self.pyobj, [self.pyobj])
        fn = self._get_function(fnty, name="PyNumber_Positive")
        return self.builder.call(fn, (obj,))

    def number_float(self, val):
        fnty = Type.function(self.pyobj, [self.pyobj])
        fn = self._get_function(fnty, name="PyNumber_Float")
        return self.builder.call(fn, [val])

    def number_invert(self, obj):
        fnty = Type.function(self.pyobj, [self.pyobj])
        fn = self._get_function(fnty, name="PyNumber_Invert")
        return self.builder.call(fn, (obj,))

    def float_as_double(self, fobj):
        fnty = Type.function(self.double, [self.pyobj])
        fn = self._get_function(fnty, name="PyFloat_AsDouble")
        return self.builder.call(fn, [fobj])

    def bool_from_bool(self, bval):
        """
        Get a Python bool from a LLVM boolean.
        """
        longval = self.builder.zext(bval, self.long)
        return self.bool_from_long(longval)

    def bool_from_long(self, ival):
        fnty = Type.function(self.pyobj, [self.long])
        fn = self._get_function(fnty, name="PyBool_FromLong")
        return self.builder.call(fn, [ival])

    def complex_from_doubles(self, realval, imagval):
        fnty = Type.function(self.pyobj, [Type.double(), Type.double()])
        fn = self._get_function(fnty, name="PyComplex_FromDoubles")
        return self.builder.call(fn, [realval, imagval])

    def complex_real_as_double(self, cobj):
        fnty = Type.function(Type.double(), [self.pyobj])
        fn = self._get_function(fnty, name="PyComplex_RealAsDouble")
        return self.builder.call(fn, [cobj])

    def complex_imag_as_double(self, cobj):
        fnty = Type.function(Type.double(), [self.pyobj])
        fn = self._get_function(fnty, name="PyComplex_ImagAsDouble")
        return self.builder.call(fn, [cobj])

    #
    # List and sequence APIs
    #

    def sequence_getslice(self, obj, start, stop):
        fnty = Type.function(self.pyobj, [self.pyobj, self.py_ssize_t,
                                          self.py_ssize_t])
        fn = self._get_function(fnty, name="PySequence_GetSlice")
        return self.builder.call(fn, (obj, start, stop))

    def sequence_tuple(self, obj):
        fnty = Type.function(self.pyobj, [self.pyobj])
        fn = self._get_function(fnty, name="PySequence_Tuple")
        return self.builder.call(fn, [obj])

    def list_new(self, szval):
        fnty = Type.function(self.pyobj, [self.py_ssize_t])
        fn = self._get_function(fnty, name="PyList_New")
        return self.builder.call(fn, [szval])

    def list_setitem(self, seq, idx, val):
        """
        Warning: Steals reference to ``val``
        """
        fnty = Type.function(Type.int(), [self.pyobj, self.py_ssize_t,
                                          self.pyobj])
        fn = self._get_function(fnty, name="PyList_SetItem")
        return self.builder.call(fn, [seq, idx, val])

    def list_getitem(self, lst, idx):
        """
        Returns a borrowed reference.
        """
        fnty = Type.function(self.pyobj, [self.pyobj, self.py_ssize_t])
        fn = self._get_function(fnty, name="PyList_GetItem")
        if isinstance(idx, int):
            idx = self.context.get_constant(types.intp, idx)
        return self.builder.call(fn, [lst, idx])

    #
    # Concrete tuple API
    #

    def tuple_getitem(self, tup, idx):
        """
        Borrow reference
        """
        fnty = Type.function(self.pyobj, [self.pyobj, self.py_ssize_t])
        fn = self._get_function(fnty, name="PyTuple_GetItem")
        idx = self.context.get_constant(types.intp, idx)
        return self.builder.call(fn, [tup, idx])

    def tuple_pack(self, items):
        fnty = Type.function(self.pyobj, [self.py_ssize_t], var_arg=True)
        fn = self._get_function(fnty, name="PyTuple_Pack")
        n = self.context.get_constant(types.intp, len(items))
        args = [n]
        args.extend(items)
        return self.builder.call(fn, args)

    def tuple_size(self, tup):
        fnty = Type.function(self.py_ssize_t, [self.pyobj])
        fn = self._get_function(fnty, name="PyTuple_Size")
        return self.builder.call(fn, [tup])

    def tuple_new(self, count):
        fnty = Type.function(self.pyobj, [Type.int()])
        fn = self._get_function(fnty, name='PyTuple_New')
        return self.builder.call(fn, [self.context.get_constant(types.int32,
                                                                count)])

    def tuple_setitem(self, tuple_val, index, item):
        """
        Steals a reference to `item`.
        """
        fnty = Type.function(Type.int(), [self.pyobj, Type.int(), self.pyobj])
        setitem_fn = self._get_function(fnty, name='PyTuple_SetItem')
        index = self.context.get_constant(types.int32, index)
        self.builder.call(setitem_fn, [tuple_val, index, item])

    #
    # Concrete set API
    #

    def set_new(self, iterable=None):
        if iterable is None:
            iterable = self.get_null_object()
        fnty = Type.function(self.pyobj, [self.pyobj])
        fn = self._get_function(fnty, name="PySet_New")
        return self.builder.call(fn, [iterable])

    def set_add(self, set, value):
        fnty = Type.function(Type.int(), [self.pyobj, self.pyobj])
        fn = self._get_function(fnty, name="PySet_Add")
        return self.builder.call(fn, [set, value])

    #
    # GIL APIs
    #

    def gil_ensure(self):
        """
        Ensure the GIL is acquired.
        The returned value must be consumed by gil_release().
        """
        gilptrty = Type.pointer(self.gil_state)
        fnty = Type.function(Type.void(), [gilptrty])
        fn = self._get_function(fnty, "numba_gil_ensure")
        gilptr = cgutils.alloca_once(self.builder, self.gil_state)
        self.builder.call(fn, [gilptr])
        return gilptr

    def gil_release(self, gil):
        """
        Release the acquired GIL by gil_ensure().
        Must be paired with a gil_ensure().
        """
        gilptrty = Type.pointer(self.gil_state)
        fnty = Type.function(Type.void(), [gilptrty])
        fn = self._get_function(fnty, "numba_gil_release")
        return self.builder.call(fn, [gil])

    def save_thread(self):
        """
        Release the GIL and return the former thread state
        (an opaque non-NULL pointer).
        """
        fnty = Type.function(self.voidptr, [])
        fn = self._get_function(fnty, name="PyEval_SaveThread")
        return self.builder.call(fn, [])

    def restore_thread(self, thread_state):
        """
        Restore the given thread state by reacquiring the GIL.
        """
        fnty = Type.function(Type.void(), [self.voidptr])
        fn = self._get_function(fnty, name="PyEval_RestoreThread")
        self.builder.call(fn, [thread_state])


    #
    # Other APIs (organize them better!)
    #

    def import_module_noblock(self, modname):
        fnty = Type.function(self.pyobj, [self.cstring])
        fn = self._get_function(fnty, name="PyImport_ImportModuleNoBlock")
        return self.builder.call(fn, [modname])

    def call_function_objargs(self, callee, objargs):
        fnty = Type.function(self.pyobj, [self.pyobj], var_arg=True)
        fn = self._get_function(fnty, name="PyObject_CallFunctionObjArgs")
        args = [callee] + list(objargs)
        args.append(self.context.get_constant_null(types.pyobject))
        return self.builder.call(fn, args)

    def call(self, callee, args, kws):
        fnty = Type.function(self.pyobj, [self.pyobj] * 3)
        fn = self._get_function(fnty, name="PyObject_Call")
        return self.builder.call(fn, (callee, args, kws))

    def object_istrue(self, obj):
        fnty = Type.function(Type.int(), [self.pyobj])
        fn = self._get_function(fnty, name="PyObject_IsTrue")
        return self.builder.call(fn, [obj])

    def object_not(self, obj):
        fnty = Type.function(Type.int(), [self.pyobj])
        fn = self._get_function(fnty, name="PyObject_Not")
        return self.builder.call(fn, [obj])

    def object_richcompare(self, lhs, rhs, opstr):
        """
        Refer to Python source Include/object.h for macros definition
        of the opid.
        """
        ops = ['<', '<=', '==', '!=', '>', '>=']
        if opstr in ops:
            opid = ops.index(opstr)
            fnty = Type.function(self.pyobj, [self.pyobj, self.pyobj, Type.int()])
            fn = self._get_function(fnty, name="PyObject_RichCompare")
            lopid = self.context.get_constant(types.int32, opid)
            return self.builder.call(fn, (lhs, rhs, lopid))
        elif opstr == 'is':
            bitflag = self.builder.icmp(lc.ICMP_EQ, lhs, rhs)
            return self.from_native_value(bitflag, types.boolean)
        elif opstr == 'is not':
            bitflag = self.builder.icmp(lc.ICMP_NE, lhs, rhs)
            return self.from_native_value(bitflag, types.boolean)
        else:
            raise NotImplementedError("Unknown operator {op!r}".format(
                op=opstr))

    def iter_next(self, iterobj):
        fnty = Type.function(self.pyobj, [self.pyobj])
        fn = self._get_function(fnty, name="PyIter_Next")
        return self.builder.call(fn, [iterobj])

    def object_getiter(self, obj):
        fnty = Type.function(self.pyobj, [self.pyobj])
        fn = self._get_function(fnty, name="PyObject_GetIter")
        return self.builder.call(fn, [obj])

    def object_getattr_string(self, obj, attr):
        cstr = self.context.insert_const_string(self.module, attr)
        fnty = Type.function(self.pyobj, [self.pyobj, self.cstring])
        fn = self._get_function(fnty, name="PyObject_GetAttrString")
        return self.builder.call(fn, [obj, cstr])

    def object_getattr(self, obj, attr):
        fnty = Type.function(self.pyobj, [self.pyobj, self.pyobj])
        fn = self._get_function(fnty, name="PyObject_GetAttr")
        return self.builder.call(fn, [obj, attr])

    def object_setattr_string(self, obj, attr, val):
        cstr = self.context.insert_const_string(self.module, attr)
        fnty = Type.function(Type.int(), [self.pyobj, self.cstring, self.pyobj])
        fn = self._get_function(fnty, name="PyObject_SetAttrString")
        return self.builder.call(fn, [obj, cstr, val])

    def object_setattr(self, obj, attr, val):
        fnty = Type.function(Type.int(), [self.pyobj, self.pyobj, self.pyobj])
        fn = self._get_function(fnty, name="PyObject_SetAttr")
        return self.builder.call(fn, [obj, attr, val])

    def object_delattr_string(self, obj, attr):
        # PyObject_DelAttrString() is actually a C macro calling
        # PyObject_SetAttrString() with value == NULL.
        return self.object_setattr_string(obj, attr, self.get_null_object())

    def object_delattr(self, obj, attr):
        # PyObject_DelAttr() is actually a C macro calling
        # PyObject_SetAttr() with value == NULL.
        return self.object_setattr(obj, attr, self.get_null_object())

    def object_getitem(self, obj, key):
        fnty = Type.function(self.pyobj, [self.pyobj, self.pyobj])
        fn = self._get_function(fnty, name="PyObject_GetItem")
        return self.builder.call(fn, (obj, key))

    def object_setitem(self, obj, key, val):
        fnty = Type.function(Type.int(), [self.pyobj, self.pyobj, self.pyobj])
        fn = self._get_function(fnty, name="PyObject_SetItem")
        return self.builder.call(fn, (obj, key, val))

    def string_as_string(self, strobj):
        fnty = Type.function(self.cstring, [self.pyobj])
        if PYVERSION >= (3, 0):
            fname = "PyUnicode_AsUTF8"
        else:
            fname = "PyString_AsString"
        fn = self._get_function(fnty, name=fname)
        return self.builder.call(fn, [strobj])

    def string_from_string_and_size(self, string, size):
        fnty = Type.function(self.pyobj, [self.cstring, self.py_ssize_t])
        if PYVERSION >= (3, 0):
            fname = "PyUnicode_FromStringAndSize"
        else:
            fname = "PyString_FromStringAndSize"
        fn = self._get_function(fnty, name=fname)
        return self.builder.call(fn, [string, size])

    def string_from_string(self, string):
        fnty = Type.function(self.pyobj, [self.cstring])
        if PYVERSION >= (3, 0):
            fname = "PyUnicode_FromString"
        else:
            fname = "PyString_FromString"
        fn = self._get_function(fnty, name=fname)
        return self.builder.call(fn, [string])

    def bytes_from_string_and_size(self, string, size):
        fnty = Type.function(self.pyobj, [self.cstring, self.py_ssize_t])
        if PYVERSION >= (3, 0):
            fname = "PyBytes_FromStringAndSize"
        else:
            fname = "PyString_FromStringAndSize"
        fn = self._get_function(fnty, name=fname)
        return self.builder.call(fn, [string, size])

    def object_str(self, obj):
        fnty = Type.function(self.pyobj, [self.pyobj])
        fn = self._get_function(fnty, name="PyObject_Str")
        return self.builder.call(fn, [obj])

    def make_none(self):
        obj = self._get_object("Py_None")
        self.incref(obj)
        return obj

    def borrow_none(self):
        obj = self._get_object("Py_None")
        return obj

    def sys_write_stdout(self, fmt, *args):
        fnty = Type.function(Type.void(), [self.cstring], var_arg=True)
        fn = self._get_function(fnty, name="PySys_WriteStdout")
        return self.builder.call(fn, (fmt,) + args)

    def object_dump(self, obj):
        """
        Dump a Python object on C stderr.  For debugging purposes.
        """
        fnty = Type.function(Type.void(), [self.pyobj])
        fn = self._get_function(fnty, name="_PyObject_Dump")
        return self.builder.call(fn, (obj,))

    # ------ utils -----

    def _get_object(self, name):
        try:
            gv = self.module.get_global_variable_named(name)
        except LLVMException:
            gv = self.module.add_global_variable(self.pyobj, name)
        return self.builder.load(gv)

    def _get_function(self, fnty, name):
        return self.module.get_or_insert_function(fnty, name=name)

    def alloca_obj(self):
        return self.builder.alloca(self.pyobj)

    def print_object(self, obj):
        strobj = self.object_str(obj)
        cstr = self.string_as_string(strobj)
        fmt = self.context.insert_const_string(self.module, "%s")
        self.sys_write_stdout(fmt, cstr)
        self.decref(strobj)

    def print_string(self, text):
        fmt = self.context.insert_const_string(self.module, text)
        self.sys_write_stdout(fmt)

    def get_null_object(self):
        return Constant.null(self.pyobj)

    def return_none(self):
        none = self.make_none()
        self.builder.ret(none)

    def list_pack(self, items):
        n = len(items)
        seq = self.list_new(self.context.get_constant(types.intp, n))
        not_null = cgutils.is_not_null(self.builder, seq)
        with cgutils.if_likely(self.builder, not_null):
            for i in range(n):
                idx = self.context.get_constant(types.intp, i)
                self.incref(items[i])
                self.list_setitem(seq, idx, items[i])
        return seq

    def to_native_arg(self, obj, typ):
        if isinstance(typ, types.Record):
            # Generate a dummy integer type that has the size of Py_buffer
            dummy_py_buffer_type = Type.int(_helperlib.py_buffer_size * 8)
            # Allocate the Py_buffer
            py_buffer = cgutils.alloca_once(self.builder, dummy_py_buffer_type)

            # Zero-fill the py_buffer. where the obj field in Py_buffer is NULL
            # PyBuffer_Release has no effect.
            zeroed_buffer = lc.Constant.null(dummy_py_buffer_type)
            self.builder.store(zeroed_buffer, py_buffer)

            buf_as_voidptr = self.builder.bitcast(py_buffer, self.voidptr)
            ptr = self.extract_record_data(obj, buf_as_voidptr)

            with cgutils.if_unlikely(self.builder,
                                     cgutils.is_null(self.builder, ptr)):
                self.builder.ret(ptr)

            ltyp = self.context.get_value_type(typ)
            val = cgutils.init_record_by_ptr(self.builder, ltyp, ptr)

            def dtor():
                self.release_record_buffer(buf_as_voidptr)

        else:
            val = self.to_native_value(obj, typ)

            def dtor():
                pass

        return val, dtor

    def to_native_value(self, obj, typ):
        if isinstance(typ, types.Object) or typ == types.pyobject:
            return obj

        elif typ == types.boolean:
            istrue = self.object_istrue(obj)
            zero = Constant.null(istrue.type)
            return self.builder.icmp(lc.ICMP_NE, istrue, zero)

        elif typ in types.unsigned_domain:
            longobj = self.number_long(obj)
            ullval = self.long_as_ulonglong(longobj)
            self.decref(longobj)
            return self.builder.trunc(ullval,
                                      self.context.get_argument_type(typ))

        elif typ in types.signed_domain:
            longobj = self.number_long(obj)
            llval = self.long_as_longlong(longobj)
            self.decref(longobj)
            return self.builder.trunc(llval,
                                      self.context.get_argument_type(typ))

        elif typ == types.float32:
            fobj = self.number_float(obj)
            fval = self.float_as_double(fobj)
            self.decref(fobj)
            return self.builder.fptrunc(fval,
                                        self.context.get_argument_type(typ))

        elif typ == types.float64:
            fobj = self.number_float(obj)
            fval = self.float_as_double(fobj)
            self.decref(fobj)
            return fval

        elif typ in (types.complex128, types.complex64):
            cplxcls = self.context.make_complex(types.complex128)
            cplx = cplxcls(self.context, self.builder)
            pcplx = cplx._getpointer()
            ok = self.complex_adaptor(obj, pcplx)
            failed = cgutils.is_false(self.builder, ok)

            with cgutils.if_unlikely(self.builder, failed):
                self.builder.ret(self.get_null_object())

            if typ == types.complex64:
                c64cls = self.context.make_complex(typ)
                c64 = c64cls(self.context, self.builder)
                freal = self.context.cast(self.builder, cplx.real,
                                          types.float64, types.float32)
                fimag = self.context.cast(self.builder, cplx.imag,
                                          types.float64, types.float32)
                c64.real = freal
                c64.imag = fimag
                return c64._getvalue()
            else:
                return cplx._getvalue()

        elif isinstance(typ, types.NPDatetime):
            val = self.extract_np_datetime(obj)
            return val

        elif isinstance(typ, types.NPTimedelta):
            val = self.extract_np_timedelta(obj)
            return val

        elif isinstance(typ, types.Array):
            return self.to_native_array(typ, obj)

        elif isinstance(typ, types.Optional):
            isnone = self.builder.icmp(lc.ICMP_EQ, obj, self.borrow_none())
            with cgutils.ifelse(self.builder, isnone) as (then, orelse):
                with then:
                    noneval = self.context.make_optional_none(self.builder, typ.type)
                    ret = cgutils.alloca_once(self.builder, noneval.type)
                    self.builder.store(noneval, ret)

                with orelse:
                    val = self.to_native_value(obj, typ.type)
                    just = self.context.make_optional_value(self.builder,
                                                            typ.type, val)
                    self.builder.store(just, ret)
            return ret

        raise NotImplementedError(typ)

    def from_native_return(self, val, typ):
        return self.from_native_value(val, typ)

    def from_native_value(self, val, typ):
        if typ == types.pyobject:
            return val

        elif typ == types.boolean:
            longval = self.builder.zext(val, self.long)
            return self.bool_from_long(longval)

        elif typ in types.unsigned_domain:
            ullval = self.builder.zext(val, self.ulonglong)
            return self.long_from_ulonglong(ullval)

        elif typ in types.signed_domain:
            ival = self.builder.sext(val, self.longlong)
            return self.long_from_longlong(ival)

        elif typ == types.float32:
            dbval = self.builder.fpext(val, self.double)
            return self.float_from_double(dbval)

        elif typ == types.float64:
            return self.float_from_double(val)

        elif typ == types.complex128:
            cmplxcls = self.context.make_complex(typ)
            cval = cmplxcls(self.context, self.builder, value=val)
            return self.complex_from_doubles(cval.real, cval.imag)

        elif typ == types.complex64:
            cmplxcls = self.context.make_complex(typ)
            cval = cmplxcls(self.context, self.builder, value=val)
            freal = self.context.cast(self.builder, cval.real,
                                      types.float32, types.float64)
            fimag = self.context.cast(self.builder, cval.imag,
                                      types.float32, types.float64)
            return self.complex_from_doubles(freal, fimag)

        elif isinstance(typ, types.NPDatetime):
            return self.create_np_datetime(val, typ.unit_code)

        elif isinstance(typ, types.NPTimedelta):
            return self.create_np_timedelta(val, typ.unit_code)

        elif typ == types.none:
            ret = self.make_none()
            return ret

        elif isinstance(typ, types.Optional):
            return self.from_native_return(val, typ.type)

        elif isinstance(typ, types.Array):
            return self.from_native_array(typ, val)

        elif isinstance(typ, types.Record):
            # Note we will create a copy of the record
            # This is the only safe way.
            pdata = cgutils.get_record_data(self.builder, val)
            size = Constant.int(Type.int(), pdata.type.pointee.count)
            ptr = self.builder.bitcast(pdata, Type.pointer(Type.int(8)))
            # Note: this will only work for CPU mode
            #       The following requires access to python object
            dtype_addr = Constant.int(self.py_ssize_t, id(typ.dtype))
            dtypeobj = dtype_addr.inttoptr(self.pyobj)
            return self.recreate_record(ptr, size, dtypeobj)

        elif isinstance(typ, (types.Tuple, types.UniTuple)):
            return self.from_tuple(typ, val)

        raise NotImplementedError(typ)

    def to_native_array(self, typ, ary):
        # TODO check matching dtype.
        #      currently, mismatching dtype will still work and causes
        #      potential memory corruption
        voidptr = Type.pointer(Type.int(8))
        nativearycls = self.context.make_array(typ)
        nativeary = nativearycls(self.context, self.builder)
        aryptr = nativeary._getpointer()
        ptr = self.builder.bitcast(aryptr, voidptr)
        errcode = self.numba_array_adaptor(ary, ptr)
        failed = cgutils.is_not_null(self.builder, errcode)
        with cgutils.if_unlikely(self.builder, failed):
            # TODO
            self.builder.unreachable()
        return self.builder.load(aryptr)

    def from_native_array(self, typ, ary):
        assert assume.return_argument_array_only
        nativearycls = self.context.make_array(typ)
        nativeary = nativearycls(self.context, self.builder, value=ary)
        parent = nativeary.parent
        self.incref(parent)
        return parent

    def from_tuple(self, typ, val):
        tuple_val = self.tuple_new(typ.count)

        for i, dtype in enumerate(typ):
            item = self.builder.extract_value(val, i)
            obj = self.from_native_value(item,  dtype)
            self.tuple_setitem(tuple_val, i, obj)

        return tuple_val

    def numba_array_adaptor(self, ary, ptr):
        voidptr = Type.pointer(Type.int(8))
        fnty = Type.function(Type.int(), [self.pyobj, voidptr])
        fn = self._get_function(fnty, name="numba_adapt_ndarray")
        fn.args[0].add_attribute(lc.ATTR_NO_CAPTURE)
        fn.args[1].add_attribute(lc.ATTR_NO_CAPTURE)
        return self.builder.call(fn, (ary, ptr))

    def complex_adaptor(self, cobj, cmplx):
        fnty = Type.function(Type.int(), [self.pyobj, cmplx.type])
        fn = self._get_function(fnty, name="numba_complex_adaptor")
        return self.builder.call(fn, [cobj, cmplx])

    def extract_record_data(self, obj, pbuf):
        fnty = Type.function(self.voidptr, [self.pyobj,
                                                         self.voidptr])
        fn = self._get_function(fnty, name="numba_extract_record_data")
        return self.builder.call(fn, [obj, pbuf])

    def release_record_buffer(self, pbuf):
        fnty = Type.function(Type.void(), [self.voidptr])
        fn = self._get_function(fnty, name="numba_release_record_buffer")
        return self.builder.call(fn, [pbuf])

    def extract_np_datetime(self, obj):
        fnty = Type.function(Type.int(64), [self.pyobj])
        fn = self._get_function(fnty, name="numba_extract_np_datetime")
        return self.builder.call(fn, [obj])

    def extract_np_timedelta(self, obj):
        fnty = Type.function(Type.int(64), [self.pyobj])
        fn = self._get_function(fnty, name="numba_extract_np_timedelta")
        return self.builder.call(fn, [obj])

    def create_np_datetime(self, val, unit_code):
        unit_code = Constant.int(Type.int(), unit_code)
        fnty = Type.function(self.pyobj, [Type.int(64), Type.int()])
        fn = self._get_function(fnty, name="numba_create_np_datetime")
        return self.builder.call(fn, [val, unit_code])

    def create_np_timedelta(self, val, unit_code):
        unit_code = Constant.int(Type.int(), unit_code)
        fnty = Type.function(self.pyobj, [Type.int(64), Type.int()])
        fn = self._get_function(fnty, name="numba_create_np_timedelta")
        return self.builder.call(fn, [val, unit_code])

    def recreate_record(self, pdata, size, dtypeaddr):
        fnty = Type.function(self.pyobj, [Type.pointer(Type.int(8)),
                                          Type.int(), self.pyobj])
        fn = self._get_function(fnty, name="numba_recreate_record")
        return self.builder.call(fn, [pdata, size, dtypeaddr])

    def string_from_constant_string(self, string):
        cstr = self.context.insert_const_string(self.module, string)
        sz = self.context.get_constant(types.intp, len(string))
        return self.string_from_string_and_size(cstr, sz)
