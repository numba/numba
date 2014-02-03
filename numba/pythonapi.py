from __future__ import print_function, division, absolute_import
from llvm.core import Type, Constant
import llvm.core as lc
import llvm.ee as le
from llvm import LLVMException
from numba.config import PYVERSION
import numba.ctypes_support as ctypes
from numba import types, utils, cgutils, _numpyadapt, _helperlib

_PyNone = ctypes.c_ssize_t(id(None))


class NativeError(RuntimeError):
    pass


@utils.runonce
def fix_python_api():
    """
    Execute once to install special symbols into the LLVM symbol table
    """
    le.dylib_add_symbol("Py_None", ctypes.addressof(_PyNone))
    le.dylib_add_symbol("NumbaArrayAdaptor", _numpyadapt.get_ndarray_adaptor())
    le.dylib_add_symbol("NumbaComplexAdaptor",
                        _helperlib.get_complex_adaptor())
    le.dylib_add_symbol("NumbaNativeError", id(NativeError))
    le.dylib_add_symbol("PyExc_NameError", id(NameError))


class PythonAPI(object):
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
        self.long = Type.int(ctypes.sizeof(ctypes.c_long) * 8)
        self.ulonglong = Type.int(ctypes.sizeof(ctypes.c_ulonglong) * 8)
        self.longlong = self.ulonglong
        self.double = Type.double()
        self.py_ssize_t = self.context.get_value_type(types.intp)
        self.cstring = Type.pointer(Type.int(8))

    # ------ Python API -----

    def incref(self, obj):
        fnty = Type.function(Type.void(), [self.pyobj])
        fn = self._get_function(fnty, name="Py_IncRef")
        self.builder.call(fn, [obj])

    def decref(self, obj):
        fnty = Type.function(Type.void(), [self.pyobj])
        fn = self._get_function(fnty, name="Py_DecRef")
        self.builder.call(fn, [obj])

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

    def dict_getitem_string(self, dic, name):
        """Returns a borrowed reference
        """
        fnty = Type.function(self.pyobj, [self.pyobj, self.cstring])
        fn = self._get_function(fnty, name="PyDict_GetItemString")
        cstr = self.context.insert_const_string(self.module, name)
        return self.builder.call(fn, [dic, cstr])

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
        return self.builder.call(fn, (exctype, msg))

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

    def long_from_long(self, ival):
        fnty = Type.function(self.pyobj, [self.long])
        fn = self._get_function(fnty, name="PyLong_FromLong")
        return self.builder.call(fn, [ival])

    def long_from_ssize_t(self, ival):
        fnty = Type.function(self.pyobj, [self.py_ssize_t])
        fn = self._get_function(fnty, name="PyLong_FromSsize_t")
        return self.builder.call(fn, [ival])

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

    def long_from_ulonglong(self, numobj):
        fnty = Type.function(self.pyobj, [self.ulonglong])
        fn = self._get_function(fnty, name="PyLong_FromUnsignedLongLong")
        return self.builder.call(fn, [numobj])

    def long_from_longlong(self, numobj):
        fnty = Type.function(self.pyobj, [self.ulonglong])
        fn = self._get_function(fnty, name="PyLong_FromLongLong")
        return self.builder.call(fn, [numobj])

    def _get_number_operator(self, name):
        fnty = Type.function(self.pyobj, [self.pyobj, self.pyobj])
        fn = self._get_function(fnty, name="PyNumber_%s" % name)
        return fn

    def number_add(self, lhs, rhs):
        fn = self._get_number_operator("Add")
        return self.builder.call(fn, [lhs, rhs])

    def number_subtract(self, lhs, rhs):
        fn = self._get_number_operator("Subtract")
        return self.builder.call(fn, [lhs, rhs])

    def number_multiply(self, lhs, rhs):
        fn = self._get_number_operator("Multiply")
        return self.builder.call(fn, [lhs, rhs])

    def number_divide(self, lhs, rhs):
        assert PYVERSION < (3, 0)
        fn = self._get_number_operator("Divide")
        return self.builder.call(fn, [lhs, rhs])

    def number_truedivide(self, lhs, rhs):
        fn = self._get_number_operator("TrueDivide")
        return self.builder.call(fn, [lhs, rhs])

    def number_floordivide(self, lhs, rhs):
        fn = self._get_number_operator("FloorDivide")
        return self.builder.call(fn, [lhs, rhs])

    def number_remainder(self, lhs, rhs):
        fn = self._get_number_operator("Remainder")
        return self.builder.call(fn, [lhs, rhs])

    def number_lshift(self, lhs, rhs):
        fn = self._get_number_operator("Lshift")
        return self.builder.call(fn, [lhs, rhs])

    def number_rshift(self, lhs, rhs):
        fn = self._get_number_operator("Rshift")
        return self.builder.call(fn, [lhs, rhs])

    def number_and(self, lhs, rhs):
        fn = self._get_number_operator("And")
        return self.builder.call(fn, [lhs, rhs])

    def number_or(self, lhs, rhs):
        fn = self._get_number_operator("Or")
        return self.builder.call(fn, [lhs, rhs])

    def number_xor(self, lhs, rhs):
        fn = self._get_number_operator("Xor")
        return self.builder.call(fn, [lhs, rhs])

    def number_power(self, lhs, rhs):
        fnty = Type.function(self.pyobj, [self.pyobj] * 3)
        fn = self._get_function(fnty, "PyNumber_Power")
        return self.builder.call(fn, [lhs, rhs, self.borrow_none()])

    def number_negative(self, obj):
        fnty = Type.function(self.pyobj, [self.pyobj])
        fn = self._get_function(fnty, name="PyNumber_Negative")
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
        opid = ops.index(opstr)
        assert 0 <= opid < len(ops)
        fnty = Type.function(self.pyobj, [self.pyobj, self.pyobj, Type.int()])
        fn = self._get_function(fnty, name="PyObject_RichCompare")
        lopid = self.context.get_constant(types.int32, opid)
        return self.builder.call(fn, (lhs, rhs, lopid))

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

    def object_getitem(self, obj, key):
        fnty = Type.function(self.pyobj, [self.pyobj, self.pyobj])
        fn = self._get_function(fnty, name="PyObject_GetItem")
        return self.builder.call(fn, (obj, key))

    def object_setitem(self, obj, key, val):
        fnty = Type.function(Type.int(), [self.pyobj, self.pyobj, self.pyobj])
        fn = self._get_function(fnty, name="PyObject_SetItem")
        return self.builder.call(fn, (obj, key, val))

    def sequence_getslice(self, obj, start, stop):
        fnty = Type.function(self.pyobj, [self.pyobj, self.py_ssize_t,
                                          self.py_ssize_t])
        fn = self._get_function(fnty, name="PySequence_GetSlice")
        return self.builder.call(fn, (obj, start, stop))

    def string_as_string(self, strobj):
        fnty = Type.function(self.cstring, [self.pyobj])
        if PYVERSION >= (3, 0):
            fname = "PyUnicode_AsUTF8"
        else:
            fname = "PyString_AsString"
        fn = self._get_function(fnty, name=fname)
        return self.builder.call(fn, [strobj])

    def string_from_string_and_size(self, string):
        fnty = Type.function(self.pyobj, [self.cstring, self.py_ssize_t])
        if PYVERSION >= (3, 0):
            fname = "PyUnicode_FromStringAndSize"
        else:
            fname = "PyString_FromStringAndSize"
        fn = self._get_function(fnty, name=fname)
        cstr = self.context.insert_const_string(self.module, string)
        sz = self.context.get_constant(types.intp, len(string))
        return self.builder.call(fn, [cstr, sz])

    def object_str(self, obj):
        fnty = Type.function(self.pyobj, [self.pyobj])
        fn = self._get_function(fnty, name="PyObject_Str")
        return self.builder.call(fn, [obj])

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

    def dict_new(self):
        fnty = Type.function(self.pyobj, ())
        fn = self._get_function(fnty, name="PyDict_New")
        return self.builder.call(fn, ())

    def dict_setitem_string(self, dictobj, name, valobj):
        fnty = Type.function(Type.int(), (self.pyobj, self.cstring,
                                          self.pyobj))
        fn = self._get_function(fnty, name="PyDict_SetItemString")
        cstr = self.context.insert_const_string(self.module, name)
        return self.builder.call(fn, (dictobj, cstr, valobj))

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
        fmt = self.context.insert_const_string(self.module, "%s\n")
        self.sys_write_stdout(fmt, cstr)
        self.decref(strobj)

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

    def to_native_arg(self, obj, typ):
        return self.to_native_value(obj, typ)

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
            pcplx = cplx._getvalue()
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

        elif isinstance(typ, types.Array):
            return self.to_native_array(typ, obj)

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

        elif typ == types.none:
            ret = self.make_none()
            return ret

        elif isinstance(typ, types.Optional):
            return self.from_native_return(val, typ.type)

        raise NotImplementedError(typ)

    def to_native_array(self, typ, ary):
        # TODO check matching dtype.
        #      currently, mismatching dtype will still work and causes
        #      potential memory corruption
        voidptr = Type.pointer(Type.int(8))
        nativearycls = self.context.make_array(typ)
        nativeary = nativearycls(self.context, self.builder)
        aryptr = nativeary._getvalue()
        ptr = self.builder.bitcast(aryptr, voidptr)
        errcode = self.numba_array_adaptor(ary, ptr)
        failed = cgutils.is_not_null(self.builder, errcode)
        with cgutils.if_unlikely(self.builder, failed):
            # TODO
            self.builder.unreachable()
        return aryptr

    def numba_array_adaptor(self, ary, ptr):
        voidptr = Type.pointer(Type.int(8))
        fnty = Type.function(Type.int(), [self.pyobj, voidptr])
        fn = self._get_function(fnty, name="NumbaArrayAdaptor")
        fn.args[0].add_attribute(lc.ATTR_NO_CAPTURE)
        fn.args[1].add_attribute(lc.ATTR_NO_CAPTURE)
        return self.builder.call(fn, (ary, ptr))

    def complex_adaptor(self, cobj, cmplx):
        fnty = Type.function(Type.int(), [self.pyobj, cmplx.type])
        fn = self._get_function(fnty, name="NumbaComplexAdaptor")
        return self.builder.call(fn, [cobj, cmplx])

    def get_module_dict_symbol(self):
        md_pymod = cgutils.MetadataKeyStore(self.module, "python.module")
        pymodname = ".pymodule.dict." + md_pymod.get()

        try:
            gv = self.module.get_global_variable_named(name=pymodname)
        except LLVMException:
            gv = self.module.add_global_variable(self.pyobj.pointee,
                                                 name=pymodname)
        return gv

    def get_module_dict(self):
        return self.get_module_dict_symbol()
        # return self.builder.load(gv)

    def raise_native_error(self, msg):
        cstr = self.context.insert_const_string(self.module, msg)
        self.err_set_string(self.native_error_type, cstr)

    @property
    def native_error_type(self):
        name = "NumbaNativeError"
        try:
            return self.module.get_global_variable_named(name)
        except LLVMException:
            return self.module.add_global_variable(self.pyobj.pointee,
                                                  name=name)

    def raise_missing_global_error(self, name):
        msg = "global name '%s' is not defined" % name
        cstr = self.context.insert_const_string(self.module, msg)
        self.err_set_string(self.name_error_type, cstr)

    @property
    def name_error_type(self):
        name = "PyExc_NameError"
        try:
            return self.module.get_global_variable_named(name)
        except LLVMException:
            return self.module.add_global_variable(self.pyobj.pointee,
                                                   name=name)
