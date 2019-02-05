# -*- coding: utf-8 -*-
"""
Support for CFFI. Allows checking whether objects are CFFI functions and
obtaining the pointer and numba signature.
"""

from types import BuiltinFunctionType
import ctypes
import re
from functools import partial
import numpy as np
import llvmlite.ir as ir

from numba import types
from numba import numpy_support
from numba import cgutils
from numba.errors import TypingError
from numba.targets import imputils
from numba.datamodel import models
from numba.datamodel import default_manager, models
from numba.typeconv import Conversion
from numba.pythonapi import box, unbox, NativeValue
from numba.core.typing import templates
from numba.np import numpy_support


try:
    import cffi
    ffi = cffi.FFI()
except ImportError:
    ffi = None

SUPPORTED = ffi is not None
_ool_func_types = {}
_ool_func_ptr = {}
_ool_libraries = {}
_ffi_instances = set()


def is_ffi_instance(obj):
    # Compiled FFI modules have a member, ffi, which is an instance of
    # CompiledFFI, which behaves similarly to an instance of cffi.FFI. In
    # order to simplify handling a CompiledFFI object, we treat them as
    # if they're cffi.FFI instances for typing and lowering purposes.
    try:
        return obj in _ffi_instances or isinstance(obj, cffi.FFI)
    except TypeError: # Unhashable type possible
        return False

def is_ffi_lib(lib):
    # we register libs on register_module call
    return lib in _ool_libraries

def is_cffi_func(obj):
    """Check whether the obj is a CFFI function"""
    try:
        return ffi.typeof(obj).kind == 'function'
    except TypeError:
        try:
            return obj.__name__ in _ool_func_types
        except:
            return False

def is_cffi_struct(obj):
    """
    Check whether the obj is a CFFI struct or a pointer to one
    """
    try:
        t = ffi.typeof(obj)
        if t.kind == 'pointer':
            return t.item in _type_map()
        elif t.kind == 'struct':
            return t in _type_map()
    except TypeError:
        return False

def get_func_pointer(cffi_func):
    """
    Get a pointer to the underlying function for a CFFI function as an
    integer.
    """
    if cffi_func.__name__ in _ool_func_ptr:
        return _ool_func_ptr[cffi_func.__name__]
    return int(ffi.cast("uintptr_t", cffi_func))

def get_struct_pointer(cffi_struct_ptr):
    """
    Convert struct pointer to integer
    """
    return int(ffi.cast("uintptr_t", cffi_struct_ptr))


_cached_type_map = None

def _type_map():
    """
    Lazily compute type map, as calling ffi.typeof() involves costly
    parsing of C code...
    """
    global _cached_type_map
    if _cached_type_map is None:
        _cached_type_map = {
            ffi.typeof('bool') :                types.boolean,
            ffi.typeof('char') :                types.char,
            ffi.typeof('short') :               types.short,
            ffi.typeof('int') :                 types.intc,
            ffi.typeof('long') :                types.long_,
            ffi.typeof('long long') :           types.longlong,
            ffi.typeof('unsigned char') :       types.uchar,
            ffi.typeof('unsigned short') :      types.ushort,
            ffi.typeof('unsigned int') :        types.uintc,
            ffi.typeof('unsigned long') :       types.ulong,
            ffi.typeof('unsigned long long') :  types.ulonglong,
            ffi.typeof('int8_t') :              types.char,
            ffi.typeof('uint8_t') :             types.uchar,
            ffi.typeof('int16_t') :             types.short,
            ffi.typeof('uint16_t') :            types.ushort,
            ffi.typeof('int32_t') :             types.intc,
            ffi.typeof('uint32_t') :            types.uintc,
            ffi.typeof('int64_t') :             types.longlong,
            ffi.typeof('uint64_t') :            types.ulonglong,
            ffi.typeof('float') :               types.float_,
            ffi.typeof('double') :              types.double,
            ffi.typeof('ssize_t') :             types.intp,
            ffi.typeof('size_t') :              types.uintp,
            ffi.typeof('void') :                types.void,
        }
    return _cached_type_map

def map_type(cffi_type, use_record_dtype=False):
    """
    Map CFFI type to numba type.

    Parameters
    ----------
    cffi_type:
        The CFFI type to be converted.
    use_record_dtype: bool (default: False)
        When True, struct types are mapped to a NumPy Record dtype.

    """
    primed_map_type = partial(map_type, use_record_dtype=use_record_dtype)
    kind = getattr(cffi_type, 'kind', '')
    if kind == 'union':
        raise TypeError("No support for CFFI union")
    elif kind == 'function':
        if cffi_type.ellipsis:
            raise TypeError("vararg function is not supported")
        restype = primed_map_type(cffi_type.result)
        argtypes = [primed_map_type(arg) for arg in cffi_type.args]
        return templates.signature(restype, *argtypes)
    elif kind == 'pointer':
        pointee = cffi_type.item
        if pointee.kind == 'void':
            return types.voidptr
        else:
            try:
                return CFFIPointer(primed_map_type(pointee))
            except TypeError:
                return types.voidptr
    elif kind == 'array':
        dtype = primed_map_type(cffi_type.item)
        nelem = cffi_type.length
        return types.NestedArray(dtype=dtype, shape=(nelem,))
    elif kind == 'struct':
        if use_record_dtype:
            return map_struct_to_record_dtype(cffi_type)
        else:
            return map_struct_to_numba_type(cffi_type)
    else:
        result = _type_map().get(cffi_type)
        if result is None:
            raise TypeError(cffi_type)
        return result

struct_registry = imputils.Registry()

class StructInstanceModel(models.StructModel):
    def __init__(self, dmm, fe_typ):
        cls_data_ty = types.ClassDataType(fe_typ)
        # MemInfoPointer uses the `dtype` attribute to traverse for nested
        # NRT MemInfo.  Since we handle nested NRT MemInfo ourselves,
        # we will replace provide MemInfoPointer with an opaque type
        # so that it does not raise exception for nested meminfo.
        dtype = types.Opaque('Opaque.' + str(cls_data_ty))
        members = [
            ('meminfo', types.MemInfoPointer(dtype)),
            ('data', types.CPointer(cls_data_ty)),
        ]
        super(StructInstanceModel, self).__init__(dmm, fe_typ, members)

class CFFIPointer(types.CPointer):
    def __init__(self, dtype):
        super(CFFIPointer, self).__init__(dtype)
        self.name = '<ffi>' + self.name

    def can_convert_to(self, typingctx, other):
        if isinstance(other, types.CPointer):
            return typingctx.can_convert_to(self.dtype, other.dtype)
        elif isinstance(other, types.RawPointer):
            return Conversion.unsafe

    def can_convert_from(self, typingctx, other):
        if isinstance(other, types.CPointer):
            return typingctx.can_convert_from(self.dtype, other.dtype)
        elif isinstance(other, types.RawPointer):
            return Conversion.unsafe

    def get_pointer(self, struct_ptr):
        return get_struct_pointer(struct_ptr)


class StructInstanceDataModel(models.StructModel):
    def __init__(self, dmm, fe_typ):
        members = [(_mangle_attr(k), v) for k, v in fe_typ.struct.items()]
        super(StructInstanceDataModel, self).__init__(dmm, fe_typ, members)

class CFFIPointerModel(models.PointerModel):
    pass

class StructInstanceType(types.Type):
    def __init__(self, cname):
        name = cname.replace(" ", "_")
        self.struct = {}
        super(StructInstanceType, self).__init__(name)

    def can_convert_to(self, typingctx, other):
        if other == types.voidptr:
            return Conversion.safe

    def can_convert_from(self, typeingctx, other):
        if other == types.voidptr:
            return Conversion.safe

class StructDataType(types.Type):
    """
    Internal only.
    Represents the data of the instance.  The representation of
    ClassInstanceType contains a pointer to a ClassDataType which represents
    a C structure that contains all the data fields of the class instance.
    """
    def __init__(self, classtyp):
        self.class_type = classtyp
        name = "data.{0}".format(self.class_type.name)
        super(StructDataType, self).__init__(name)


@imputils.lower_cast(CFFIPointer, CFFIPointer)
@imputils.lower_cast(CFFIPointer, types.voidptr)
def voidptr_to_cffipointer(context, builder, fromty, toty, val):
    res = builder.bitcast(val, context.get_data_type(toty))
    return imputils.impl_ret_untracked(context, builder, toty, res)

def _mangle_attr(name):
    """
    Mangle attributes.
    The resulting name does not startswith an underscore '_'.
    """
    return 'm_' + name


default_manager.register(StructInstanceType, StructInstanceModel)
default_manager.register(StructInstanceType, StructInstanceDataModel)
default_manager.register(CFFIPointer, CFFIPointerModel)


@struct_registry.lower_getattr_generic(StructInstanceType)
def field_impl(context, builder, typ, value, attr):
    """
    Generic getattr for cffi.StructInstanceType
    """
    if attr in typ.struct:
        inst = context.make_helper(builder, typ, value=value)
        data_pointer = inst
        data = context.make_data_helper(builder, typ.get_data_type(),
            ref=data_pointer)
        return imputils.impl_ret_borrowed(context, builder, typ.struct[attr],
            getattr(data, _mangle_attr(attr)))

@templates.infer_getattr
class StructAttribute(templates.AttributeTemplate):
    key = StructInstanceType

    def generic_resolve(self, instance, attr):
        if attr in instance.struct:
            return instance.struct[attr]


def map_struct_to_numba_type(cffi_type):
    """
    Convert a cffi type to a numba StructType
    """
    struct_type = StructInstanceType(cffi_type.cname)
    for k, v in cffi_type.fields:
        if v.bitshift != -1:
            msg = "field {!r} has bitshift, this is not supported"
            raise ValueError(msg.format(k))
        if v.flags != 0:
            msg = "field {!r} has flags, this is not supported"
            raise ValueError(msg.format(k))
        if v.bitsize != -1:
            msg = "field {!r} has bitsize, this is not supported"
            raise ValueError(msg.format(k))
        struct_type.struct[k] = map_type(v.type, use_record_dtype=False)
    return struct_type

def map_struct_to_record_dtype(cffi_type):
    """Convert a cffi type into a NumPy Record dtype
    """
    fields = {
            'names': [],
            'formats': [],
            'offsets': [],
            'itemsize': ffi.sizeof(cffi_type),
    }
    is_aligned = True
    for k, v in cffi_type.fields:
        # guard unsupport values
        if v.bitshift != -1:
            msg = "field {!r} has bitshift, this is not supported"
            raise ValueError(msg.format(k))
        if v.flags != 0:
            msg = "field {!r} has flags, this is not supported"
            raise ValueError(msg.format(k))
        if v.bitsize != -1:
            msg = "field {!r} has bitsize, this is not supported"
            raise ValueError(msg.format(k))
        dtype = numpy_support.as_dtype(
            map_type(v.type, use_record_dtype=True),
        )
        fields['names'].append(k)
        fields['formats'].append(dtype)
        fields['offsets'].append(v.offset)
        # Check alignment
        is_aligned &= (v.offset % dtype.alignment == 0)

    return numpy_support.from_dtype(np.dtype(fields, align=is_aligned))


def make_function_type(cffi_func, use_record_dtype=False):
    """
    Return a Numba type for the given CFFI function pointer.
    """
    if isinstance(cffi_func, str):
        cffi_type = _ool_func_types.get(cffi_func)
    else:
        cffi_type = _ool_func_types.get(cffi_func.__name__) or ffi.typeof(cffi_func)
    sig = map_type(cffi_type, use_record_dtype=use_record_dtype)
    return types.ExternalFunctionPointer(sig, get_pointer=get_func_pointer)

def get_struct_type(cffi_struct):
    """
    Return a Numba type for the given CFFI struct
    """
    t = ffi.typeof(cffi_struct)
    if t.kind == "pointer":
        return CFFIPointer(_type_map()[t.item])

    return _type_map()[t]

@box(CFFIPointer)
def struct_instance_box(typ, val, c):
    import ipdb; ipdb.set_trace()

@unbox(CFFIPointer)
def struct_instance_ptr_unbox(typ, val, c):
    ptrty = c.context.data_model_manager[typ].get_value_type()
    ret = c.builder.alloca(ptrty)
    ser = c.pyapi.serialize_object(typ.get_pointer)
    get_pointer = c.pyapi.unserialize(ser)
    intobj = c.pyapi.call_function_objargs(get_pointer, (val, ))
    c.pyapi.decref(get_pointer)
    with cgutils.if_likely(c.builder, cgutils.is_not_null(c.builder, intobj)):
        ptr = c.pyapi.long_as_voidptr(intobj)
        c.pyapi.decref(intobj)
        c.builder.store(c.builder.bitcast(ptr, ptrty), ret)
    return NativeValue(c.builder.load(ret), is_error=c.pyapi.c_api_error())

class CFFILibrary(types.Type):
    def __init__(self, lib):
        self._func_names = set(f for f in dir(lib) \
            if isinstance(getattr(lib, f), BuiltinFunctionType))
        self._lib_name = re.match(r"<Lib object for '([^']+)'>", str(lib)).group(1)
        name = 'cffi_lib<{}>'.format(self._lib_name)
        super(CFFILibrary, self).__init__(name)

    def has_func(self, func_name):
        return func_name in self._func_names

    def get_func_pointer(self, func_name):
        if not func_name in self._func_names:
            raise AttributeError("Function {} is not present in the library {}".format(
                func_name, self._lib_name))
        return _ool_func_ptr[func_name]


class CFFILibraryDataModel(models.OpaqueModel):
    pass


default_manager.register(CFFILibrary, CFFILibraryDataModel)


registry = templates.Registry()


@registry.register_attr
class CFFILibAttr(templates.AttributeTemplate):
    key = CFFILibrary

    def generic_resolve(self, instance, attr):
        if instance.has_func(attr):
            return make_function_type(attr)


@imputils.lower_constant(CFFILibrary)
def lower_cffi_library(context, builder, ty, pyval):
    return context.get_dummy_value()
    # return ir.Constant(ir.IntType(8).as_pointer(), None)


@imputils.lower_getattr_generic(CFFILibrary)
def lower_get_func(context, builder, typ, value, attr):
    pyapi = context.get_python_api(builder)
    if not typ.has_func(attr):
        raise AttributeError("Function {} is not present in the library {}".format(
                func_name, self._lib_name))
    func_typ = make_function_type(attr)
    # Call get_func_pointer() on the object to get the raw pointer value
    ptrty = context.get_function_pointer_type(func_typ)
    ret = cgutils.alloca_once_value(builder,
                                    ir.Constant(ptrty, None),
                                    name='fnptr')
    ser = pyapi.serialize_object(typ)
    runtime_typ = pyapi.unserialize(ser)
    attr_cstr = context.insert_const_string(builder.module, attr)
    intobj = pyapi.call_method(runtime_typ, "get_func_pointer",
        [pyapi.string_from_string(attr_cstr)])
    with cgutils.if_likely(builder,
                            cgutils.is_not_null(builder, intobj)):
        ptr = pyapi.long_as_voidptr(intobj)
        pyapi.decref(intobj)
        builder.store(builder.bitcast(ptr, ptrty), ret)
    return builder.load(ret)

@registry.register
class FFI_from_buffer(templates.AbstractTemplate):
    key = 'ffi.from_buffer'

    def generic(self, args, kws):
        if kws or len(args) != 1:
            return
        [ary] = args
        if not isinstance(ary, types.Buffer):
            raise TypingError("from_buffer() expected a buffer object, got %s"
                              % (ary,))
        if ary.layout not in ('C', 'F'):
            raise TypingError("from_buffer() unsupported on non-contiguous buffers (got %s)"
                              % (ary,))
        if ary.layout != 'C' and ary.ndim > 1:
            raise TypingError("from_buffer() only supports multidimensional arrays with C layout (got %s)"
                              % (ary,))
        ptr = types.CPointer(ary.dtype)
        return templates.signature(ptr, ary)

@registry.register_attr
class FFIAttribute(templates.AttributeTemplate):
    key = types.ffi

    def resolve_from_buffer(self, ffi):
        return types.BoundFunction(FFI_from_buffer, types.ffi)

def register_type(cffi_type, numba_type):
    """
    Add typing for a given CFFI type to the typemap
    """
    tm = _type_map()
    tm[cffi_type] = numba_type

def register_module(mod):
    """
    Add typing for all functions in an out-of-line CFFI module to the typemap
    """
    _ool_libraries[mod.lib] = CFFILibrary(mod.lib)
    for t in mod.ffi.list_types()[0]:
        cffi_type = mod.ffi.typeof(t)
        register_type(cffi_type, map_struct_to_numba_type(cffi_type))
    for f in dir(mod.lib):
        f = getattr(mod.lib, f)
        if isinstance(f, BuiltinFunctionType):
            _ool_func_types[f.__name__] = mod.ffi.typeof(f)
            addr = mod.ffi.addressof(mod.lib, f.__name__)
            _ool_func_ptr[f.__name__] = int(mod.ffi.cast("uintptr_t", addr))
        _ffi_instances.add(mod.ffi)

