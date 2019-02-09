# -*- coding: utf-8 -*-
"""
Support for CFFI. Allows checking whether objects are CFFI functions and
obtaining the pointer and numba signature.
"""
from __future__ import print_function, division, absolute_import

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
from numba import typing
from numba.errors import TypingError
from numba.targets import imputils
from numba.datamodel import models, register_default
from numba.datamodel import default_manager, models
from numba.typeconv import Conversion
from numba.pythonapi import box, unbox, NativeValue
from . import templates
from .cffi_utils import *


class FFI_new(types.BoundFunction):
    def __init__(self, ffi):
        # we inherit from bound function to avoid
        # implementing dummy lowering for getattr
        # we also don't call super, since we don't
        # actually share the logic
        self.ffi = ffi
        self.typing_key = "ffi.new"
        self.this = ffi
        name = "({}).new".format(ffi)
        super(types.BoundFunction, self).__init__(name)

    def copy(self):
        return type(self)(self.ffi)

    def get_call_type(self, context, args, kws):
        if len(args) > 1:
            raise NotImplementedError
        cffi_type = self.ffi.ffi.typeof(args[0].literal_value)

        if cffi_type.kind == "pointer":
            pointee = cffi_type.item
            ret_type = types.CFFIPointer(cffi_type_map()[pointee], owning=True)
        elif cffi_type.kind == "array":
            length = cffi_type.length
            pointee = cffi_type.item
            ret_type = types.CFFIArrayType(
                cffi_type_map()[pointee], length, owning=True
            )
        else:
            raise ValueError("Can only allocate memory for pointer and array types")
        return templates.signature(ret_type, *args)


default_manager.register(FFI_new, models.handle_bound_function)


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

    def get_field(self, builder, target, field):
        pos = self.get_field_pos(field)
        return builder.extract_value(target, pos)


class CFFIPointerModel(models.PointerModel):
    def get_field(self, builder, target, field):
        pos = self._pointee_model.get_field_pos(field)
        return builder.gep(target, [cgutils.int32_t(0), cgutils.int32_t(pos)])

    def set_field(self, builder, target, field, value):
        pos = self._pointee_model.get_field_pos(field)
        ptr = builder.gep(target, [cgutils.int32_t(0), cgutils.int32_t(pos)])
        return builder.store(value, ptr)


class CFFIArrayModel(models.PointerModel):
    def get_item(self, builder, target, index):
        pass

    def set_item(self, builder, target, index):
        pass


def _mangle_attr(name):
    """
    Mangle attributes.
    The resulting name does not startswith an underscore '_'.
    """
    return "m_" + name


default_manager.register(types.CFFIStructInstanceType, CFFIStructInstanceModel)
default_manager.register(types.CFFIPointer, CFFIPointerModel)
default_manager.register(types.CFFIArrayType, CFFIPointerModel)
default_manager.register(types.CFFIStructRefType, CFFIPointerModel)


registry = templates.Registry()


@imputils.lower_getattr_generic(types.CFFIStructInstanceType)
def field_impl(context, builder, typ, value, attr):
    """
    Generic getattr for cffi.CFFIStructInstanceType
    """
    if attr in typ.struct:
        ddm = context.data_model_manager[typ]
        data = ddm.get_field(builder, value, attr)
        return imputils.impl_ret_borrowed(context, builder, typ.struct[attr], data)


@imputils.lower_getattr_generic(types.CFFIPointer)
def pointer_field_impl(context, builder, typ, value, attr):
    """
    Generic getattr for cffi.CFFIStructInstanceType pointer
    """
    pointee = typ.dtype
    if attr in pointee.struct:
        ddm = context.data_model_manager[typ]
        ret = ddm.get_field(builder, value, attr)
        return imputils.impl_ret_borrowed(
            context, builder, typ.dtype.struct[attr], builder.load(ret)
        )


@imputils.lower_getattr_generic(types.CFFIStructRefType)
def ref_field_impl(context, builder, typ, value, attr):
    """
    Generic getattr for cffi ref type
    """
    pointee = typ.dtype
    if attr in pointee.struct:
        ddm = context.data_model_manager[typ]
        ret = ddm.get_field(builder, value, attr)
        return imputils.impl_ret_borrowed(
            context, builder, typ.dtype.struct[attr], builder.load(ret)
        )


# getattr on a cffi pointer should return a pointer to the element
# to correspond to cffi's behaviour
@registry.register_global(operator.getitem)
class CFFIPointerGetitem(templates.AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        ptr, idx = args
        if isinstance(ptr, types.CFFIPointer) and isinstance(idx, types.Integer):
            return templates.signature(
                types.CFFIStructRefType(ptr),
                ptr,
                idx,
            )


@registry.register_global(len)
class CFFIArrayLength(templates.AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        ptr = args[0]
        if isinstance(ptr, types.CFFIArrayType):
            return templates.signature(types.int64, ptr)


@imputils.lower_builtin(len, types.CFFIArrayType)
def len_cffiarray(context, builder, sig, args):
    res = cgutils.intp_t(sig.args[0].length)
    return imputils.impl_ret_untracked(context, builder, sig.return_type, res)

@imputils.lower_builtin('getiter', types.CFFIArrayType)
def getiter_cffiarray(context, builder, sig, args):
    iterobj = context.make_helper(builder, sig.return_type)

    zero = context.get_constant(types.intp, 0)
    indexptr = cgutils.alloca_once_value(builder, zero)

    iterobj.index = indexptr
    iterobj.array = args[0]

    if context.enable_nrt:
        context.nrt.incref(builder, sig.args[0], args[0])

    res = iterobj._getvalue()

    return imputils.impl_ret_new_ref(context, builder, sig.return_type, res)


@imputils.lower_builtin('iternext', types.CFFIIteratorType)
@imputils.iternext_impl
def iternext_cffiarray(context, builder, sig, args, result):
    iterty = sig.args[0]
    iter_ = args[0]
    containerty = iterty.container

    iterobj = context.make_helper(builder, iterty, value=iter_)
    length = cgutils.intp_t(containerty.length)
    index = builder.load(iterobj.index)
    is_valid = builder.icmp_unsigned('<', index, length)
    result.set_valid(is_valid)

    with builder.if_then(is_valid):
        value = builder.gep(iterobj.array, [index])
        result.yield_(value)
        nindex = cgutils.increment_index(builder, index)
        builder.store(nindex, iterobj.index)

@imputils.lower_builtin(operator.getitem, types.CFFIPointer, types.Integer)
def getitem_cffipointer(context, builder, sig, args):
    base_ptr, idx = args
    res = builder.gep(base_ptr, [idx])
    return imputils.impl_ret_borrowed(context, builder, sig.return_type, res)


@imputils.lower_setattr_generic(types.CFFIStructInstanceType)
def set_struct_field_impl(context, builder, sig, args, attr):
    """
    Generic setattr for cffi.CFFIStructInstanceType pointer
    """
    raise ValueError(
        "setfield on a struct is not implemented. Use setfield on ptr or ref type"
    )


@imputils.lower_setattr_generic(types.CFFIPointer)
def set_pointer_field_impl(context, builder, sig, args, attr):
    """
    Generic setattr for cffi CFFIStructInstanceType pointer
    """
    target, val = args
    targetty, valty = sig.args
    pointee = targetty.dtype
    if attr in pointee.struct:
        ddm = context.data_model_manager[targetty]
        return ddm.set_field(builder, target, attr, val)
    else:
        raise ValueError("Cannot setattr {} on {}".format(attr, sig))


@imputils.lower_cast(types.CFFIStructRefType, types.CFFIStructInstanceType)
def cast_ref_to_struct(context, builder, fromty, toty, val):
    res = builder.load(val)
    return imputils.impl_ret_borrowed(context, builder, toty, res)


@registry.register_attr
class StructAttribute(templates.AttributeTemplate):
    key = types.CFFIStructInstanceType

    def generic_resolve(self, instance, attr):
        if attr in instance.struct:
            return instance.struct[attr]


@registry.register_attr
class cffi_poitner_attr(templates.AttributeTemplate):
    key = types.CFFIPointer

    def generic_resolve(self, instance, attr):
        pointee = instance.dtype
        if attr in pointee.struct:
            return pointee.struct[attr]


@imputils.lower_constant(types.CFFIPointer)
def lower_const_cffi_pointer(context, builder, ty, pyval):
    ptrty = context.get_value_type(ty)
    ptrval = context.add_dynamic_addr(builder, ty.get_pointer(pyval), info=str(pyval))
    return builder.bitcast(ptrval, ptrty)


@box(types.CFFIPointer)
def struct_instance_box(typ, val, c):
    ser = c.pyapi.serialize_object(struct_from_ptr)
    struct_from_ptr_runtime = c.pyapi.unserialize(ser)
    pointee = typ.dtype

    hash_ = cffi_types_cache.get_type_hash(pointee.cffi_type)
    hash_ = cgutils.intp_t(hash_)
    hash_ = c.pyapi.long_from_longlong(hash_)
    cast_val = c.builder.ptrtoint(val, cgutils.intp_t)
    struct_addr = c.pyapi.long_from_ssize_t(cast_val)
    owning = c.pyapi.bool_from_bool(cgutils.bool_t(typ.owning))
    args = [hash_, struct_addr, owning]
    if isinstance(typ, types.CFFIArrayType):
        args.append(c.pyapi.long_from_ssize_t(cgutils.intp_t(typ.length)))
    return c.pyapi.call_function_objargs(struct_from_ptr_runtime, args)


# this is the layout ob the CFFI's CDataObject
# see https://bitbucket.org/cffi/cffi/src/86332166be5b05759060f81e0acacbdebdd3075b/c/_cffi_backend.c#_cffi_backend.c-216
cffi_cdata_type = ir.LiteralStructType(
    [
        ir.ArrayType(cgutils.int8_t, 16),  # 16-byte PyObject_HEAD
        cgutils.voidptr_t,  # CTypeDescrObject* ctypes
        cgutils.intp_t.as_pointer(),  # cdata
        cgutils.voidptr_t,  # PyObject *c_weakreflist;
    ]
)


@unbox(types.CFFIPointer)
def struct_instance_ptr_unbox(typ, val, c):
    # this is roughtly 10 times faster than going back to python and
    # calling ffi.cast('uintptr_t')
    ptrty = c.context.data_model_manager[typ].get_value_type()
    ret = c.builder.alloca(ptrty)
    cffi_data_ptr = c.builder.bitcast(val, cffi_cdata_type.as_pointer())
    intptr = c.builder.extract_value(c.builder.load(cffi_data_ptr), 2)
    c.builder.store(c.builder.bitcast(intptr, ptrty), ret)
    return NativeValue(c.builder.load(ret))


@registry.register_attr
class CFFILibAttr(templates.AttributeTemplate):
    key = types.CFFILibraryType

    def generic_resolve(self, instance, attr):
        if instance.has_func(attr):
            return make_function_type(attr)


@imputils.lower_getattr_generic(types.CFFILibraryType)
def lower_get_func(context, builder, typ, value, attr):
    pyapi = context.get_python_api(builder)
    if not typ.has_func(attr):
        raise AttributeError("Function {} is not present in the library".format(attr))
    func_typ = make_function_type(attr)
    # Call get_func_pointer() on the object to get the raw pointer value
    ptrty = context.get_function_pointer_type(func_typ)
    ret = cgutils.alloca_once_value(builder, ir.Constant(ptrty, None), name="fnptr")
    # function address is constant and can't be overwritten from python
    # so we cache it
    func_addr = cgutils.intp_t(typ.get_func_pointer(attr))
    builder.store(builder.inttoptr(func_addr, ptrty), ret)
    return builder.load(ret)


class CFFINullPtrType(types.CPointer):
    def __init__(self):
        super(CFFINullPtrType, self).__init__(types.void)

    def can_convert_from(self, typeingctx, other):
        if isinstance(other, types.CFFIPointer):
            return Conversion.safe

    def can_convert_to(self, typeingctx, other):
        if isinstance(other, types.CFFIPointer):
            return Conversion.safe


@register_default(CFFINullPtrType)
class CFFINullPtrModel(models.PointerModel):
    pass


@register_default(types.CFFIIteratorType)
class CFFIIteratorModel(models.StructModel):
    def __init__(self, dmm, fe_type):
        # We use an unsigned index to avoid the cost of negative index tests.
        members = [('index', types.EphemeralPointer(types.uintp)),
                   ('array', fe_type.container)]
        super(CFFIIteratorModel, self).__init__(dmm, fe_type, members)



@imputils.lower_getattr(types.FFIType, "NULL")
def lower_ffi_null(context, builder, sig, args):
    return context.get_constant_null(CFFINullPtrType())


@registry.register
class FFI_from_buffer(templates.AbstractTemplate):
    key = "ffi.from_buffer"

    def generic(self, args, kws):
        if kws or len(args) != 1:
            return
        [ary] = args
        if not isinstance(ary, types.Buffer):
            raise TypingError("from_buffer() expected a buffer object, got %s" % (ary,))
        if ary.layout not in ("C", "F"):
            raise TypingError(
                "from_buffer() unsupported on non-contiguous buffers (got %s)" % (ary,)
            )
        if ary.layout != "C" and ary.ndim > 1:
            raise TypingError(
                "from_buffer() only supports multidimensional arrays with C layout (got %s)"
                % (ary,)
            )
        ptr = types.CPointer(ary.dtype)
        return templates.signature(ptr, ary)


@templates.infer
class FFI_new_direct(templates.AbstractTemplate):
    key = "ffi.new"

    def generic(self, args, kws):
        raise ValueError("Please call new through the ffi object (ffi.new)")


@imputils.lower_builtin("ffi.new", types.Literal)
def from_buffer(context, builder, sig, args):
    retty = context.get_value_type(sig.return_type)
    struct_size = context.get_abi_sizeof(retty.pointee)

    # if it's an array, adjust size accordingly
    if isinstance(sig.return_type, types.CFFIArrayType):
        struct_size *= sig.return_type.length

    # ffi.new allocates zero initialized memory, we do it too
    ptr = context.nrt.allocate(builder, cgutils.intp_t(struct_size))
    memset = builder.module.declare_intrinsic(
        "llvm.memset", [cgutils.int8_t.as_pointer(), cgutils.int32_t]
    )
    builder.call(
        memset,
        [ptr, cgutils.int8_t(0), cgutils.int32_t(struct_size), cgutils.bool_t(0)],
    )
    ret = builder.bitcast(ptr, retty)
    return imputils.impl_ret_untracked(context, builder, sig.return_type, ret)


@registry.register_attr
class FFIAttribute(templates.AttributeTemplate):
    key = types.FFIType

    def resolve_from_buffer(self, ffi):
        return types.BoundFunction(FFI_from_buffer, ffi)

    def resolve_NULL(self, ffi):
        return CFFINullPtrType()

    def resolve_new(self, ffi):
        return FFI_new(ffi)


@templates.infer_global(operator.ne)
@templates.infer_global(operator.eq)
class PtrCMPTemplate(templates.AbstractTemplate):
    def generic(self, args, kws):
        (ptr1, ptr2) = args
        if isinstance(ptr1, types.CPointer) and isinstance(ptr2, types.CPointer):
            return templates.signature(types.bool_, ptr1, ptr2)


@imputils.lower_builtin(operator.ne, CFFINullPtrType, types.CPointer)
def lower_null_ptr_ne_pos1(context, builder, sig, args):
    to_compare = args[1]
    int_ptr = builder.ptrtoint(to_compare, cgutils.intp_t)
    res = builder.icmp_unsigned("!=", int_ptr, cgutils.intp_t(0))
    return imputils.impl_ret_untracked(context, builder, sig.return_type, res)


@imputils.lower_builtin(operator.ne, types.CPointer, CFFINullPtrType)
def lower_null_ptr_ne_pos2(context, builder, sig, args):
    to_compare = args[0]
    int_ptr = builder.ptrtoint(to_compare, cgutils.intp_t)
    res = builder.icmp_unsigned("!=", int_ptr, cgutils.intp_t(0))
    return imputils.impl_ret_untracked(context, builder, sig.return_type, res)


@imputils.lower_builtin(operator.eq, CFFINullPtrType, types.CPointer)
def lower_null_ptr_eq_pos1(context, builder, sig, args):
    to_compare = args[1]
    int_ptr = builder.ptrtoint(to_compare, cgutils.intp_t)
    res = builder.icmp_unsigned("==", int_ptr, cgutils.intp_t(0))
    return imputils.impl_ret_untracked(context, builder, sig.return_type, res)


@imputils.lower_builtin(operator.eq, types.CPointer, CFFINullPtrType)
def lower_null_ptr_eq_pos2(context, builder, sig, args):
    to_compare = args[0]
    int_ptr = builder.ptrtoint(to_compare, cgutils.intp_t)
    res = builder.icmp_unsigned("==", int_ptr, cgutils.intp_t(0))
    return imputils.impl_ret_untracked(context, builder, sig.return_type, res)

