from llvm.core import Type, Constant
import llvm.ee as le
from llvm import LLVMException
import ctypes
from numba import types, utils

_PyNone = ctypes.c_ssize_t(id(None))


@utils.runonce
def fix_python_api():
    le.dylib_add_symbol("Py_None", ctypes.addressof(_PyNone))


class PythonAPI(object):
    def __init__(self, context, builder):
        fix_python_api()
        self.context = context
        self.builder = builder
        self.pyobj = self.context.get_argument_type(types.pyobject)
        self.long = Type.int(ctypes.sizeof(ctypes.c_long) * 8)
        self.py_ssize_t = self.context.get_value_type(types.intp)
        self.cstring = Type.pointer(Type.int(8))
        self.module = builder.basic_block.function.module

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

    def long_from_long(self, ival):
        fnty = Type.function(self.pyobj, [self.long])
        fn = self._get_function(fnty, name="PyInt_FromLong")
        return self.builder.call(fn, [ival])

    def number_as_ssize_t(self, numobj):
        fnty = Type.function(self.py_ssize_t, [self.pyobj])
        fn = self._get_function(fnty, name="PyNumber_AsSsize_t")
        return self.builder.call(fn, [numobj])

    def _get_number_operator(self, name):
        fnty = Type.function(self.pyobj, [self.pyobj, self.pyobj])
        fn = self._get_function(fnty, name="PyNumber_%s" % name)
        return fn

    def number_add(self, lhs, rhs):
        fn = self._get_number_operator("Add")
        return self.builder.call(fn, [lhs, rhs])

    def bool_from_long(self, ival):
        fnty = Type.function(self.pyobj, [self.long])
        fn = self._get_function(fnty, name="PyBool_FromLong")
        return self.builder.call(fn, [ival])

    def object_getattr_string(self, obj, attr):
        cstr = self.context.insert_const_string(self.module, attr)
        fnty = Type.function(self.pyobj, [self.pyobj, self.cstring])
        fn = self._get_function(fnty, name="PyObject_GetAttrString")
        return self.builder.call(fn, [obj, cstr])

    def string_as_string(self, strobj):
        fnty = Type.function(self.cstring, [self.pyobj])
        fn = self._get_function(fnty, name="PyString_AsString")
        return self.builder.call(fn, [strobj])

    def string_from_string_and_size(self, string):
        fnty = Type.function(self.pyobj, [self.cstring, self.py_ssize_t])
        fn = self._get_function(fnty, name="PyString_FromStringAndSize")
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
        fn = self._get_function(fnty, "PyTuple_GetItem")
        idx = self.context.get_constant(types.intp, idx)
        return self.builder.call(fn, [tup, idx])

    def make_none(self):
        obj = self._get_object("Py_None")
        self.incref(obj)
        return obj

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

    def get_null_object(self):
        return Constant.null(self.pyobj)

    def to_native_arg(self, obj, typ):
        if typ == types.pyobject:
            return obj
        elif typ == types.int32:
            ssize_val = self.number_as_ssize_t(obj)
            return self.builder.trunc(ssize_val,
                                      self.context.get_argument_type(typ))
        elif isinstance(typ, types.Array):
            return self.to_native_array(typ, obj)
        raise NotImplementedError(typ)

    def from_native_return(self, val, typ):
        if typ == types.pyobject:
            return val
        elif typ == types.boolean:
            longval = self.builder.zext(val, self.long)
            return self.bool_from_long(longval)

        elif typ == types.int32:
            longval = self.builder.sext(val, self.long)
            return self.long_from_long(longval)

        elif typ == types.none:
            ret = self.make_none()
            return ret
        raise NotImplementedError(typ)

    def to_native_array(self, typ, ary):
        nativearycls = self.context.make_array(typ)
        nativeary = nativearycls(self.context, self.builder)
        ctobj = self.object_getattr_string(ary, "ctypes")
        cdata = self.object_getattr_string(ctobj, "data")
        pyshape = self.object_getattr_string(ary, "shape")
        pystrides = self.object_getattr_string(ary, "strides")

        rawint = self.number_as_ssize_t(cdata)

        nativeary.data = self.builder.inttoptr(rawint, nativeary.data.type)

        shapeary = nativeary.shape
        strideary = nativeary.strides

        for i in range(typ.ndim):
            shape = self.tuple_getitem(pyshape, i)
            stride = self.tuple_getitem(pystrides, i)

            shapeval = self.number_as_ssize_t(shape)
            strideval = self.number_as_ssize_t(stride)

            shapeary = self.builder.insert_value(shapeary, shapeval, i)
            strideary = self.builder.insert_value(strideary, strideval, i)

        nativeary.shape = shapeary
        nativeary.strides = strideary

        self.decref(cdata)
        self.decref(pyshape)
        self.decref(pystrides)

        return nativeary._getvalue()

