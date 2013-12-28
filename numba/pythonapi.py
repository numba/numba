from llvm.core import Type, Constant
import ctypes
from numba import types


class PythonAPI(object):
    def __init__(self, context, builder):
        self.context = context
        self.builder = builder
        self.pyobj = self.context.get_argument_type(types.pyobject)
        self.long = Type.int(ctypes.sizeof(ctypes.c_long) * 8)
        self.module = builder.basic_block.function.module

    def _get_function(self, fnty, name):
        return self.module.get_or_insert_function(fnty, name=name)

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

    def alloca_obj(self):
        return self.builder.alloca(self.pyobj)

    def get_null_object(self):
        return Constant.null(self.pyobj)

    def int_as_long(self, intobj):
        fnty = Type.function(self.long, [self.pyobj])
        fn = self._get_function(fnty, name="PyInt_AsLong")
        return self.builder.call(fn, [intobj])

    def int_from_long(self, ival):
        fnty = Type.function(self.pyobj, [self.long])
        fn = self._get_function(fnty, name="PyInt_FromLong")
        return self.builder.call(fn, [ival])

    def bool_from_long(self, ival):
        fnty = Type.function(self.pyobj, [self.long])
        fn = self._get_function(fnty, name="PyBool_FromLong")
        return self.builder.call(fn, [ival])

    def to_native_arg(self, obj, typ):
        if typ == types.int32:
            longval = self.int_as_long(obj)
            return self.builder.trunc(longval,
                                      self.context.get_argument_type(typ))
        raise NotImplementedError(typ)

    def from_native_return(self, val, typ):
        if typ == types.boolean:
            longval = self.builder.zext(val, self.long)
            return self.bool_from_long(longval)
        if typ == types.int32:
            longval = self.builder.sext(val, self.long)
            return self.int_from_long(longval)
        raise NotImplementedError(typ)
