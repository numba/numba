from llvmlite import ir

from numba.core import types, cgutils

# Status codes for NpyString operations (must match C in numba/_helperlib.c)
NPYSTRING_STATUS_ERROR = -1  # Error occurred
NPYSTRING_STATUS_OK = 0      # Success, not NA
NPYSTRING_STATUS_NA = 1      # Success, is NA


def get_descr_ptr(context, builder, stringdtype_typ):
    """Return a `void*` pointer to the underlying PyArray_Descr for a
    StringDTypeType.
    """
    addr = context.add_dynamic_addr(builder, stringdtype_typ.descr_pointer,
                                    info="StringDType.descr")
    ll_voidptr = context.get_value_type(types.voidptr)
    return builder.bitcast(addr, ll_voidptr)


def alloc_packed_storage(context, builder, typ):
    """Allocate transient storage for a StringDType packed value.

    Returns (packed_storage, packed_ptr).
    """
    dm = context.data_model_manager[typ]
    data_type = dm.get_data_type()
    packed_storage = cgutils.alloca_once(builder, data_type)
    packed_ptr = builder.bitcast(packed_storage, ir.IntType(8).as_pointer())
    return packed_storage, packed_ptr


def pack_from_pyobject(
    context, builder, descr_voidptr, packed_ptr, pyapi, pyobj,
):
    """Call the pack helper to convert a Python object into packed storage.

    Returns the status (int32). Caller is responsible for decref of pyobj.
    """
    fn_ty = ir.FunctionType(ir.IntType(32), [
        context.get_value_type(types.voidptr),
        ir.IntType(8).as_pointer(),
        pyapi.pyobj,
    ])
    fn = cgutils.get_or_insert_function(builder.module, fn_ty,
                                        name="numba_stringdtype_pack")
    return builder.call(fn, [descr_voidptr, packed_ptr, pyobj])


def unpack_to_pyobject(context, builder, descr_voidptr, packed_ptr, pyapi):
    """Call the object-path unpack helper: returns a new PyObject*."""
    fn_ty = ir.FunctionType(pyapi.pyobj, [
        context.get_value_type(types.voidptr),
        ir.IntType(8).as_pointer(),
    ])
    fn = cgutils.get_or_insert_function(builder.module, fn_ty,
                                        name="numba_stringdtype_unpack")
    return builder.call(fn, [descr_voidptr, packed_ptr])


def unpack_utf8_status(context, builder, descr_voidptr, packed_ptr):
    """Call the nogil UTF-8 unpack helper and return status code only."""
    ll_i8_ptr = ir.IntType(8).as_pointer()

    fn_ty = ir.FunctionType(ir.IntType(32), [
        context.get_value_type(types.voidptr),
        ll_i8_ptr,
    ])
    fn = cgutils.get_or_insert_function(
        builder.module, fn_ty,
        name="numba_stringdtype_utf8_status",
    )
    return builder.call(fn, [descr_voidptr, packed_ptr])


def prepare_from_value(context, builder, typ, val):
    """Given a StringDType scalar `val`,
    produce (descr_voidptr, packed_ptr, descr_ptr).

    - Ensures descriptor is non-NULL (falls back to global descriptor)
    - Prepares a transient packed storage pointer suitable for C helpers
    """
    ll_voidptr = context.get_value_type(types.voidptr)
    ll_i8_ptr = ir.IntType(8).as_pointer()

    # Extract fields
    descr_ptr_typed = builder.extract_value(val, 0)
    data_value = builder.extract_value(val, 1)

    # Fill descriptor if missing
    tmp = cgutils.alloca_once_value(builder, descr_ptr_typed)
    is_null = cgutils.is_null(builder, descr_ptr_typed)
    with builder.if_then(is_null, likely=False):
        fallback = get_descr_ptr(context, builder, typ)
        fallback_cast = builder.bitcast(fallback, descr_ptr_typed.type)
        builder.store(fallback_cast, tmp)
    final_descr_ptr = builder.load(tmp)

    # Emplace data_value into temporary storage to yield i8* to packed region
    storage = cgutils.alloca_once_value(builder, data_value)
    packed_ptr = builder.bitcast(storage, ll_i8_ptr)
    descr_voidptr = builder.bitcast(final_descr_ptr, ll_voidptr)
    return descr_voidptr, packed_ptr, final_descr_ptr


def utf8_status_pair(context, builder, ty_a, val_a, ty_b, val_b):
    """Return UTF-8 load status codes for two StringDType scalars."""
    ll_voidptr = context.get_value_type(types.voidptr)
    ll_i8_ptr = ir.IntType(8).as_pointer()
    status_fn_ty = ir.FunctionType(ir.IntType(32), [
        ll_voidptr,
        ll_i8_ptr,
    ])
    status_fn = cgutils.get_or_insert_function(
        builder.module, status_fn_ty,
        name="numba_stringdtype_utf8_status",
    )

    def unpack_one(ty, val):
        descr_void, packed_ptr, _ = prepare_from_value(
            context, builder, ty, val
        )
        return builder.call(status_fn, [descr_void, packed_ptr])

    status_a = unpack_one(ty_a, val_a)
    status_b = unpack_one(ty_b, val_b)
    return status_a, status_b
