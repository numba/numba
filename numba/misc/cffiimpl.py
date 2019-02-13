"""
Implementation of some CFFI functions
"""


import operator

from llvmlite import ir

from numba.core.imputils import Registry
from numba import types
from numba.core import imputils
from numba import cgutils
from numba.core.typing.cffi_utils import (
    make_function_type,
    get_func_pointer,
    get_struct_pointer,
    get_free_ffi_func,
)
from numba.misc import arrayobj


registry = Registry()


@registry.lower("ffi.from_buffer", types.Buffer)
def from_buffer(context, builder, sig, args):
    assert len(sig.args) == 1
    assert len(args) == 1
    [fromty] = sig.args
    [val] = args
    # Type inference should have prevented passing a buffer from an
    # array to a pointer of the wrong type
    assert fromty.dtype == sig.return_type.dtype
    ary = arrayobj.make_array(fromty)(context, builder, val)
    return ary.data


@registry.lower_getattr_generic(types.CFFIStructInstanceType)
def field_impl(context, builder, typ, value, attr):
    """
    Generic getattr for cffi.CFFIStructInstanceType
    """
    if attr in typ.struct:
        ddm = context.data_model_manager[typ]
        data = ddm.get_field(builder, value, attr)
        return imputils.impl_ret_new_ref(context, builder, typ.struct[attr], data)


@registry.lower_getattr_generic(types.CFFIPointer)
def pointer_field_impl(context, builder, typ, value, attr):
    """
    Generic getattr for cffi.CFFIStructInstanceType pointer
    """
    pointee = typ.dtype
    if attr in pointee.struct:
        ddm = context.data_model_manager[typ]
        ret = ddm.get_field(builder, value, attr)
        return imputils.impl_ret_new_ref(
            context, builder, typ.dtype.struct[attr], builder.load(ret)
        )


@registry.lower_getattr_generic(types.CFFIStructRefType)
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


@registry.lower(len, types.CFFIArrayType)
def len_cffiarray(context, builder, sig, args):
    res = cgutils.intp_t(sig.args[0].length)
    return imputils.impl_ret_untracked(context, builder, sig.return_type, res)


@registry.lower("getiter", types.CFFIArrayType)
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


@registry.lower("iternext", types.CFFIIteratorType)
@imputils.iternext_impl
def iternext_cffiarray(context, builder, sig, args, result):
    iterty = sig.args[0]
    iter_ = args[0]
    containerty = iterty.container

    iterobj = context.make_helper(builder, iterty, value=iter_)
    length = cgutils.intp_t(containerty.length)
    index = builder.load(iterobj.index)
    is_valid = builder.icmp_unsigned("<", index, length)
    result.set_valid(is_valid)

    with builder.if_then(is_valid):
        value = builder.gep(iterobj.array, [index])
        result.yield_(value)
        nindex = cgutils.increment_index(builder, index)
        builder.store(nindex, iterobj.index)


@registry.lower(operator.getitem, types.CFFIPointer, types.Integer)
def getitem_cffipointer(context, builder, sig, args):
    base_ptr, idx = args
    res = builder.gep(base_ptr, [idx])
    return imputils.impl_ret_new_ref(context, builder, sig.return_type, res)


@registry.lower(operator.getitem, types.CFFIOwningType, types.Integer)
def getitem_owned_cffipointer(context, builder, sig, args):
    obj, idx = args
    obj = context.make_helper(builder, sig.args[0], value=obj)
    base_ptr = obj.data
    res = builder.gep(base_ptr, [idx])
    if context.enable_nrt:
        context.nrt.incref(builder, sig.args[0], args[0])
    return imputils.impl_ret_new_ref(context, builder, sig.return_type, res)


@registry.lower_setattr_generic(types.CFFIStructInstanceType)
def set_struct_field_impl(context, builder, sig, args, attr):
    """
    Generic setattr for cffi.CFFIStructInstanceType pointer
    """
    raise ValueError(
        "setfield on a struct is not implemented. Use setfield on ptr or ref type"
    )


@registry.lower_setattr_generic(types.CFFIPointer)
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


@registry.lower_constant(types.CFFIPointer)
def lower_const_cffi_pointer(context, builder, ty, pyval):
    ptrty = context.get_value_type(ty)
    ptrval = context.add_dynamic_addr(
        builder, get_struct_pointer(pyval), info=str(pyval)
    )
    return builder.bitcast(ptrval, ptrty)


@registry.lower_cast(types.CFFIStructRefType, types.CFFIStructInstanceType)
def cast_ref_to_struct(context, builder, fromty, toty, val):
    res = builder.load(val)
    return imputils.impl_ret_new_ref(context, builder, toty, res)


@registry.lower_cast(types.CFFIOwningType, types.CFFIPointer)
def cast_owned_to_plain(context, builder, fromty, toty, val):
    obj = context.make_helper(builder, fromty, value=val)
    res = builder.bitcast(obj.data, context.get_value_type(toty))
    return imputils.impl_ret_new_ref(context, builder, toty, res)


@registry.lower_getattr_generic(types.CFFILibraryType)
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
    if not typ.has_func(attr):
        raise AttributeError("Function {} is not present in the library".format(attr))
    func_addr = cgutils.intp_t(get_func_pointer(attr))
    builder.store(builder.inttoptr(func_addr, ptrty), ret)
    return builder.load(ret)


@registry.lower_getattr(types.FFIType, "NULL")
def lower_ffi_null(context, builder, sig, args):
    return context.get_constant_null(types.CFFINullPtrType())


@registry.lower("ffi.new", types.Literal)
def ffi_new(context, builder, sig, args):
    retty = context.get_value_type(sig.return_type)
    dataty = context.get_value_type(sig.return_type.dtype)
    struct_size = context.get_abi_sizeof(dataty)

    # if it's an array, adjust size accordingly
    if isinstance(sig.return_type, types.CFFIArrayType):
        struct_size *= sig.return_type.length

    if isinstance(sig.return_type, types.CFFIOwningType):
        ret = context.make_helper(builder, sig.return_type)
        ret.meminfo = context.nrt.meminfo_alloc(builder, cgutils.intp_t(struct_size))
        ret.data = builder.bitcast(
            context.nrt.meminfo_data(builder, ret.meminfo), dataty.as_pointer()
        )
        cgutils.memset(builder, ret.data, cgutils.intp_t(struct_size), 0)
        return imputils.impl_ret_new_ref(
            context, builder, sig.return_type, ret._getvalue()
        )
    else:
        # ffi.new allocates zero initialized memory, we do it too
        ptr = context.nrt.allocate(builder, cgutils.intp_t(struct_size))
        cgutils.memset(builder, ptr, cgutils.intp_t(struct_size), 0)
        ret = builder.bitcast(ptr, retty)
        return imputils.impl_ret_untracked(context, builder, sig.return_type, ret)


@registry.lower(operator.ne, types.CFFINullPtrType, types.CPointer)
def lower_null_ptr_ne_pos1(context, builder, sig, args):
    to_compare = args[1]
    int_ptr = builder.ptrtoint(to_compare, cgutils.intp_t)
    res = builder.icmp_unsigned("!=", int_ptr, cgutils.intp_t(0))
    return imputils.impl_ret_untracked(context, builder, sig.return_type, res)


@registry.lower(operator.ne, types.CPointer, types.CFFINullPtrType)
def lower_null_ptr_ne_pos2(context, builder, sig, args):
    to_compare = args[0]
    int_ptr = builder.ptrtoint(to_compare, cgutils.intp_t)
    res = builder.icmp_unsigned("!=", int_ptr, cgutils.intp_t(0))
    return imputils.impl_ret_untracked(context, builder, sig.return_type, res)


@registry.lower(operator.eq, types.CFFINullPtrType, types.CPointer)
def lower_null_ptr_eq_pos1(context, builder, sig, args):
    to_compare = args[1]
    int_ptr = builder.ptrtoint(to_compare, cgutils.intp_t)
    res = builder.icmp_unsigned("==", int_ptr, cgutils.intp_t(0))
    return imputils.impl_ret_untracked(context, builder, sig.return_type, res)


@registry.lower(operator.eq, types.CPointer, types.CFFINullPtrType)
def lower_null_ptr_eq_pos2(context, builder, sig, args):
    to_compare = args[0]
    int_ptr = builder.ptrtoint(to_compare, cgutils.intp_t)
    res = builder.icmp_unsigned("==", int_ptr, cgutils.intp_t(0))
    return imputils.impl_ret_untracked(context, builder, sig.return_type, res)

