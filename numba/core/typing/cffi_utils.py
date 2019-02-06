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
import operator

from numba import types
from numba import numpy_support
from numba import cgutils
from numba.errors import TypingError
from numba.targets import imputils
from numba.datamodel import models, register_default
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
_ool_libraries = {}
_ool_func_types = {}
_ool_struct_types = {}
_ool_func_ptr = {}
_ffi_instances = set()

class CFFITypeInfo(object):
    """
    Cache cffi type info
    """
    def __init__(self, ffi, cffi_type):
        self.cname = cffi_type.cname
        self.cffi_type = cffi_type
        self.cffi_ptr_t = ffi.typeof(self.cffi_type.cname + '*')

class CFFIStructTypeCache(object):
    def __init__(self):
        self.ctypes_cache = {}
        self.type_to_lib = {}

    def add_type(self, mod, cffi_type):
        self.ctypes_cache[hash((mod.lib, cffi_type))] = CFFITypeInfo(mod.ffi, cffi_type)
        self.type_to_lib[cffi_type] = mod.lib

    def get_type_by_hash(self, h):
        return self.ctypes_cache[h]

    def get_types_lib(self, cffi_type):
        return self.type_to_lib[cffi_type]

    def get_hash(self, lib, cffi_type):
        return hash((lib, cffi_type))

_cffi_types_cache = CFFIStructTypeCache()

def debug():
    return _ool_struct_types, _ool_func_types


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
    except TypeError:
        return False
    if t.kind == 'pointer':
        return t.item.kind == 'struct'
    elif t.kind == 'struct':
        return True
    else:
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
    result = _type_map().get(cffi_type)
    if result is not None:
        return result
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
            except TypeError as e:
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
        raise TypeError(cffi_type)


class CFFIStructInstanceModel(models.StructModel):
    def __init__(self, dmm, fe_typ):
        members = []
        self._field_pos = {}
        for idx, (field, member_fe_typ) in enumerate(fe_typ.struct.items()):
            members.append((field, member_fe_typ))
            self._field_pos[field] = idx
        super(CFFIStructInstanceModel, self).__init__(dmm, fe_typ, members)

    def get_field_pos(self, field):
        return self._field_pos[field]

    def get_field(self, builder, value, field):
        pos = self.get_field_pos(self, field)
        return builder.extract_value(value, pos)


class CFFIPointer(types.CPointer):
    def __init__(self, dtype):
        super(CFFIPointer, self).__init__(dtype)
        self.name = '<ffi>(' + self.name + ')'

    @staticmethod
    def get_pointer(struct_ptr):
        return get_struct_pointer(struct_ptr)


class CFFIPointerModel(models.PointerModel):
    def get_field(self, builder, value, field):
        pos = self._pointee_model.get_field_pos(field)
        return builder.gep(value, [cgutils.int32_t(0), cgutils.int32_t(pos)])


class CFFIStructInstanceType(types.Type):
    def __init__(self, cffi_type):
        self.cffi_type = cffi_type
        name = "<instance> (" + self.cffi_type.cname + ")"
        self.struct = {}
        super(CFFIStructInstanceType, self).__init__(name)

    def can_convert_to(self, typingctx, other):
        if other == types.voidptr:
            return Conversion.safe

    def can_convert_from(self, typeingctx, other):
        if other == types.voidptr:
            return Conversion.safe


def _mangle_attr(name):
    """
    Mangle attributes.
    The resulting name does not startswith an underscore '_'.
    """
    return 'm_' + name


default_manager.register(CFFIStructInstanceType, CFFIStructInstanceModel)
default_manager.register(CFFIPointer, CFFIPointerModel)


@imputils.lower_getattr_generic(CFFIStructInstanceType)
def field_impl(context, builder, typ, value, attr):
    """
    Generic getattr for cffi.CFFIStructInstanceType
    """
    if attr in typ.struct:
        ddm = context.data_model_manager[typ]
        data = ddm.get_field(builder, value, attr)
        return imputils.impl_ret_borrowed(context, builder, typ.struct[attr],
            data)

@imputils.lower_getattr_generic(CFFIPointer)
def pointer_field_impl(context, builder, typ, value, attr):
    """
    Generic getattr for cffi.CFFIStructInstanceType
    """
    pointee = typ.dtype
    if attr in pointee.struct:
        ddm = context.data_model_manager[typ]
        ret = ddm.get_field(builder, value, attr)
        return imputils.impl_ret_borrowed(context, builder, typ.dtype.struct[attr],
            builder.load(ret))


@templates.infer_getattr
class StructAttribute(templates.AttributeTemplate):
    key = CFFIStructInstanceType

    def generic_resolve(self, instance, attr):
        if attr in instance.struct:
            return instance.struct[attr]

@templates.infer_getattr
class CFFIPointerAttr(templates.AttributeTemplate):
    key = CFFIPointer

    def generic_resolve(self, instance, attr):
        pointee = instance.dtype
        if attr in pointee.struct:
            return pointee.struct[attr]


def map_struct_to_numba_type(cffi_type):
    """
    Convert a cffi type to a numba StructType
    """
    if cffi_type in _ool_struct_types:
        return _ool_struct_types[cffi_type]

    # forward declare the struct type
    forward = types.deferred_type()
    _ool_struct_types[cffi_type] = forward
    struct_type = CFFIStructInstanceType(cffi_type)
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
    forward.define(struct_type)
    _ool_struct_types[cffi_type] = struct_type
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

def struct_from_ptr(h, intptr):
    return ffi.cast(_cffi_types_cache.get_type_by_hash(h).cffi_ptr_t, intptr)



@box(CFFIPointer)
def struct_instance_box(typ, val, c):
    ser = c.pyapi.serialize_object(struct_from_ptr)
    struct_from_ptr_runtime = c.pyapi.unserialize(ser)
    pointee = typ.dtype

    h = hash((_cffi_types_cache.get_types_lib(pointee.cffi_type), pointee.cffi_type))
    h = cgutils.intp_t(h)
    h = c.pyapi.long_from_longlong(h)
    cast_val = c.builder.ptrtoint(val, cgutils.intp_t)
    struct_addr = c.pyapi.long_from_ssize_t(cast_val)
    return c.pyapi.call_function_objargs(struct_from_ptr_runtime, [h, struct_addr])

# this is the layout ob the CFFI's CDataObject
# see https://bitbucket.org/cffi/cffi/src/86332166be5b05759060f81e0acacbdebdd3075b/c/_cffi_backend.c#_cffi_backend.c-216
cffi_cdata_type = ir.LiteralStructType([
    ir.ArrayType(cgutils.int8_t, 16),  # 16-byte PyObject_HEAD
    cgutils.voidptr_t,  # CTypeDescrObject* ctypes
    cgutils.intp_t.as_pointer(),  #cdata
    cgutils.voidptr_t,  # PyObject *c_weakreflist;
])


@unbox(CFFIPointer)
def struct_instance_ptr_unbox(typ, val, c):
    # this is roughtly 10 times faster than going back to python and
    # calling ffi.cast('uintptr_t')
    ptrty = c.context.data_model_manager[typ].get_value_type()
    ret = c.builder.alloca(ptrty)
    cffi_data_ptr = c.builder.bitcast(val, cffi_cdata_type.as_pointer())
    intptr = c.builder.extract_value(c.builder.load(cffi_data_ptr), 2)
    c.builder.store(c.builder.bitcast(intptr, ptrty), ret)
    return NativeValue(c.builder.load(ret))

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


@register_default(CFFILibrary)
class CFFILibraryDataModel(models.OpaqueModel):
    pass


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
        raise AttributeError("Function {} is not present in the library".format(
                attr))
    func_typ = make_function_type(attr)
    # Call get_func_pointer() on the object to get the raw pointer value
    ptrty = context.get_function_pointer_type(func_typ)
    ret = cgutils.alloca_once_value(builder,
                                    ir.Constant(ptrty, None),
                                    name='fnptr')
    # function address is constant and can't be overwritten from python
    # so we cache it
    func_addr = cgutils.intp_t(typ.get_func_pointer(attr))
    builder.store(builder.inttoptr(func_addr, ptrty), ret)
    return builder.load(ret)


class CFFINullPtrType(types.CPointer):
    def __init__(self):
        super(CFFINullPtrType, self).__init__(types.void)

    def can_convert_from(self, typeingctx, other):
        if isinstance(other, CFFIPointer):
            return Conversion.safe

    def can_convert_to(self, typeingctx, other):
        if isinstance(other, CFFIPointer):
            return Conversion.safe



class CFFINullPtrModel(models.PointerModel):
    pass

default_manager.register(CFFINullPtrType, CFFINullPtrModel)

@imputils.lower_getattr(types.ffi, 'NULL')
def lower_ffi_null(context, builder, sig, args):
    return context.get_constant_null(CFFINullPtrType())

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

    def resolve_NULL(self, ffi):
        return CFFINullPtrType()

@templates.infer_global(operator.ne)
@templates.infer_global(operator.eq)
class PtrCMPTemplate(templates.AbstractTemplate):
    def generic(self, args, kws):
        (ptr1, ptr2) = args
        if isinstance(ptr1, types.CPointer) and isinstance(ptr2, types.CPointer):
            return templates.signature(types.bool_, ptr1, ptr2)

@imputils.lower_builtin(operator.ne, CFFINullPtrType, types.CPointer)
def lower_null_ptr_cmp_pos1(context, builder, sig, args):
    to_compare = args[1]
    int_ptr = builder.ptrtoint(to_compare, cgutils.intp_t)
    res = builder.icmp_unsigned('!=',int_ptr, cgutils.intp_t(0))
    return imputils.impl_ret_untracked(context, builder, sig.return_type, res)

@imputils.lower_builtin(operator.ne, types.CPointer, CFFINullPtrType)
def lower_null_ptr_cmp_pos2(context, builder, sig, args):
    to_compare = args[0]
    int_ptr = builder.ptrtoint(to_compare, cgutils.intp_t)
    res = builder.icmp_unsigned('!=',int_ptr, cgutils.intp_t(0))
    return imputils.impl_ret_untracked(context, builder, sig.return_type, res)

@imputils.lower_builtin(operator.eq, CFFINullPtrType, types.CPointer)
def lower_null_ptr_cmp_pos1(context, builder, sig, args):
    to_compare = args[1]
    int_ptr = builder.ptrtoint(to_compare, cgutils.intp_t)
    res = builder.icmp_unsigned('==',int_ptr, cgutils.intp_t(0))
    return imputils.impl_ret_untracked(context, builder, sig.return_type, res)

@imputils.lower_builtin(operator.eq, types.CPointer, CFFINullPtrType)
def lower_null_ptr_cmp_pos2(context, builder, sig, args):
    to_compare = args[0]
    int_ptr = builder.ptrtoint(to_compare, cgutils.intp_t)
    res = builder.icmp_unsigned('==',int_ptr, cgutils.intp_t(0))
    return imputils.impl_ret_untracked(context, builder, sig.return_type, res)



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
    if mod.lib in _ool_libraries:
        # module already registered, don't do anything
        return
    _ool_libraries[mod.lib] = CFFILibrary(mod.lib)
    for t in mod.ffi.list_types()[0]:
        cffi_type = mod.ffi.typeof(t)
        register_type(cffi_type, map_struct_to_numba_type(cffi_type))
        _cffi_types_cache.add_type(mod, cffi_type)
    for f in dir(mod.lib):
        f = getattr(mod.lib, f)
        if isinstance(f, BuiltinFunctionType):
            _ool_func_types[f.__name__] = mod.ffi.typeof(f)
            addr = mod.ffi.addressof(mod.lib, f.__name__)
            _ool_func_ptr[f.__name__] = int(mod.ffi.cast("uintptr_t", addr))
        _ffi_instances.add(mod.ffi)

