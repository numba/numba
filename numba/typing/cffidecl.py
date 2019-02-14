# -*- coding: utf-8 -*-
"""
Support for CFFI. Allows checking whether objects are CFFI functions and
obtaining the pointer and numba signature.
"""
from __future__ import print_function, division, absolute_import

import operator

from llvmlite import ir

from numba import types
from numba import cgutils
from numba.errors import TypingError
from numba.datamodel import models, register_default, default_manager
from numba.pythonapi import box, unbox, NativeValue
from numba.runtime import cffidynmod
from . import templates
from .cffi_utils import (
    cffi_type_map,
    make_function_type,
    struct_from_ptr,
    cffi_types_cache,
)


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
            ret_type = types.CFFIOwningPointerType(cffi_type_map()[pointee])
        elif cffi_type.kind == "array":
            length = cffi_type.length
            pointee = cffi_type.item
            ret_type = types.CFFIOwningArrayType(cffi_type_map()[pointee], length)
        else:
            raise ValueError("Can only allocate memory for pointer and array types")
        return templates.signature(ret_type, *args)


default_manager.register(FFI_new, models.handle_bound_function)


@register_default(types.CFFIStructInstanceType)
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


@register_default(types.CFFIPointer)
@register_default(types.CFFIArrayType)
class CFFIPointerModel(models.PointerModel):
    def get_field(self, builder, target, field):
        pos = self._pointee_model.get_field_pos(field)
        return builder.gep(target, [cgutils.int32_t(0), cgutils.int32_t(pos)])

    def set_field(self, builder, target, field, value):
        pos = self._pointee_model.get_field_pos(field)
        ptr = builder.gep(target, [cgutils.int32_t(0), cgutils.int32_t(pos)])
        return builder.store(value, ptr)


@register_default(types.CFFIOwningPointerType)
@register_default(types.CFFIOwningArrayType)
class CFFIOwningPointerModel(models.StructModel):
    def __init__(self, dmm, fe_typ):
        # opaque ptr type trick to avoid recursion in meminfo
        # see also jitclass/base.py:35
        opaque_typ = types.Opaque("Opaque." + str(fe_typ))
        members = [
            ("meminfo", types.MemInfoPointer(opaque_typ)),
            ("data", types.CFFIPointer(fe_typ.dtype)),
        ]
        self.dmodel = dmm.lookup(fe_typ.dtype)
        super(CFFIOwningPointerModel, self).__init__(dmm, fe_typ, members)

    def get_field(self, builder, target, field):
        pos = self.dmodel.get_field_pos(field)
        data = builder.extract_value(target, 1)
        return builder.gep(data, [cgutils.int32_t(0), cgutils.int32_t(pos)])

    def set_field(self, builder, target, field, value):
        pos = self.dmodel.get_field_pos(field)
        data = builder.extract_value(target, 1)
        ptr = builder.gep(data, [cgutils.int32_t(0), cgutils.int32_t(pos)])
        return builder.store(value, ptr)


@register_default(types.CFFIStructRefType)
class CFFIRefTypeModel(models.PointerModel):
    def get_field(self, builder, target, field):
        pos = self._pointee_model.get_field_pos(field)
        return builder.gep(target, [cgutils.int32_t(0), cgutils.int32_t(pos)])

    def set_field(self, builder, target, field, value):
        pos = self._pointee_model.get_field_pos(field)
        ptr = builder.gep(target, [cgutils.int32_t(0), cgutils.int32_t(pos)])
        return builder.store(value, ptr)


@register_default(types.CFFIOwningStructRefType)
class CFFIOwningRefTypeModel(models.StructModel):
    def __init__(self, dmm, fe_typ):
        # opaque ptr type trick to avoid recursion in meminfo
        # see also jitclass/base.py:35
        opaque_typ = types.Opaque("Opaque." + str(fe_typ))
        members = [
            ("meminfo", types.MemInfoPointer(opaque_typ)),
            ("data", types.CFFIPointer(fe_typ.dtype)),
        ]
        self.dmodel = dmm.lookup(fe_typ.dtype)
        super(CFFIOwningRefTypeModel, self).__init__(dmm, fe_typ, members)

    def get_field(self, builder, target, field):
        pos = self.dmodel.get_field_pos(field)
        data = builder.extract_value(target, 1)
        return builder.gep(data, [cgutils.int32_t(0), cgutils.int32_t(pos)])

    def set_field(self, builder, target, field, value):
        pos = self.dmodel.get_field_pos(field)
        data = builder.extract_value(target, 1)
        ptr = builder.gep(data, [cgutils.int32_t(0), cgutils.int32_t(pos)])
        return builder.store(value, ptr)


@register_default(types.CFFINullPtrType)
class CFFINullPtrModel(models.PointerModel):
    pass


@register_default(types.CFFIIteratorType)
class CFFIIteratorModel(models.StructModel):
    def __init__(self, dmm, fe_type):
        # We use an unsigned index to avoid the cost of negative index tests.
        members = [
            ("index", types.EphemeralPointer(types.uintp)),
            ("array", fe_type.container),
        ]
        super(CFFIIteratorModel, self).__init__(dmm, fe_type, members)


registry = templates.Registry()


@registry.register_global(operator.getitem)
class CFFIPointerGetitem(templates.AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        ptr, idx = args
        if isinstance(ptr, types.CFFIPointer):
            if not isinstance(idx, types.Integer):
                raise TypeError("Only integer indexing is supported on CFFI pointers")

            if isinstance(ptr.dtype, types.CFFIStructInstanceType):
                if isinstance(ptr, types.CFFIOwningType):
                    return templates.signature(
                        types.CFFIOwningStructRefType(ptr), ptr, idx
                    )
                else:
                    return templates.signature(types.CFFIStructRefType(ptr), ptr, idx)
            else:
                return templates.signature(ptr.dtype, ptr, idx)


@registry.register_global(len)
class CFFIArrayLength(templates.AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        ptr = args[0]
        if isinstance(ptr, types.CFFIArrayType):
            return templates.signature(types.int64, ptr)


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


@registry.register_attr
class CFFILibAttr(templates.AttributeTemplate):
    key = types.CFFILibraryType

    def generic_resolve(self, instance, attr):
        if instance.has_func(attr):
            return make_function_type(attr)


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


@registry.register_attr
class FFIAttribute(templates.AttributeTemplate):
    key = types.FFIType

    def resolve_from_buffer(self, ffi):
        return types.BoundFunction(FFI_from_buffer, ffi)

    def resolve_NULL(self, ffi):
        return types.CFFINullPtrType()

    def resolve_new(self, ffi):
        return FFI_new(ffi)


@templates.infer_global(operator.ne)
@templates.infer_global(operator.eq)
class PtrCMPTemplate(templates.AbstractTemplate):
    def generic(self, args, kws):
        (ptr1, ptr2) = args
        if isinstance(ptr1, types.CPointer) and isinstance(ptr2, types.CPointer):
            return templates.signature(types.bool_, ptr1, ptr2)


@box(types.CFFIPointer)
def struct_instance_box(typ, val, c):
    ser = c.pyapi.serialize_object(struct_from_ptr)
    struct_from_ptr_runtime = c.pyapi.unserialize(ser)
    pointee = typ.dtype

    hash_ = cffi_types_cache.get_type_hash(pointee)
    hash_ = cgutils.intp_t(hash_)
    hash_ = c.pyapi.long_from_longlong(hash_)
    if isinstance(typ, types.CFFIOwningType):
        obj = c.context.make_helper(c.builder, typ, value=val)
        data_val = c.builder.ptrtoint(obj.data, cgutils.intp_t)
        free_val = c.builder.ptrtoint(obj.meminfo, cgutils.intp_t)
        owned = True
    else:
        data_val = c.builder.ptrtoint(val, cgutils.intp_t)
        free_val = cgutils.intp_t(0)
        owned = False
    data_addr = c.pyapi.long_from_ssize_t(data_val)
    free_addr = c.pyapi.long_from_ssize_t(free_val)
    owning = c.pyapi.bool_from_bool(cgutils.bool_t(owned))
    args = [hash_, data_addr, free_addr, owning]
    if isinstance(typ, types.CFFIArrayType):
        args.append(c.pyapi.long_from_ssize_t(cgutils.intp_t(typ.length)))
    return c.pyapi.call_function_objargs(struct_from_ptr_runtime, args)


# this is the layout ob the CFFI's CDataObject
# see https://bitbucket.org/cffi/cffi/src/86332166be5b05759060f81e0acacbdebdd3075b/c/_cffi_backend.c#_cffi_backend.c-216
cffi_cdata_type = ir.LiteralStructType(
    [
        ir.ArrayType(cgutils.int8_t, 16),  # 16-byte PyObject_HEAD
        cgutils.voidptr_t,  # CTypeDescrObject* ctypes
        cgutils.voidptr_t,  # cdata
        cgutils.voidptr_t,  # PyObject *c_weakreflist;
    ]
)


@unbox(types.CFFIPointer)
def struct_instance_ptr_unbox(typ, val, c):
    # this is roughtly 10 times faster than going back to python and
    # calling ffi.cast('uintptr_t')
    ptrty = c.context.data_model_manager[typ].get_value_type()
    ret = c.builder.alloca(ptrty)
    cdataptr = cffidynmod.get_cffi_pointer(c.builder, val)
    c.builder.store(c.builder.bitcast(cdataptr, ptrty), ret)
    return NativeValue(c.builder.load(ret))
