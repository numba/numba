
from llvmlite import ir

from numba.core import types, cgutils
from numba.core.pythonapi import box, unbox, NativeValue

from contextlib import ExitStack


@box(types.Record)
def box_record(typ, val, c):
    # Note we will create a copy of the record
    # This is the only safe way.
    size = ir.Constant(ir.IntType(32), val.type.pointee.count)
    ptr = c.builder.bitcast(val, ir.PointerType(ir.IntType(8)))
    return c.pyapi.recreate_record(ptr, size, typ.dtype, c.env_manager)


@unbox(types.Record)
def unbox_record(typ, obj, c):
    buf = c.pyapi.alloca_buffer()
    ptr = c.pyapi.extract_record_data(obj, buf)
    is_error = cgutils.is_null(c.builder, ptr)

    ltyp = c.context.get_value_type(typ)
    val = c.builder.bitcast(ptr, ltyp)

    def cleanup():
        c.pyapi.release_buffer(buf)
    return NativeValue(val, cleanup=cleanup, is_error=is_error)


@box(types.DType)
def box_dtype(typ, val, c):
    from numba.np import numpy_support
    np_dtype = numpy_support.as_dtype(typ.dtype)
    return c.pyapi.unserialize(c.pyapi.serialize_object(np_dtype))


@unbox(types.DType)
def unbox_dtype(typ, val, c):
    return NativeValue(c.context.get_dummy_value())


# Original implementation at:
# https://github.com/numba/numba/issues/4499#issuecomment-1063138477
@unbox(types.NumPyRandomBitGeneratorType)
def unbox_numpy_random_bitgenerator(typ, obj, c):
    """
    The bit_generator instance has a `.ctypes` attr which is a namedtuple
    with the following members (types):
    * state_address (Python int)
    * state (ctypes.c_void_p)
    * next_uint64 (ctypes.CFunctionType instance)
    * next_uint32 (ctypes.CFunctionType instance)
    * next_double (ctypes.CFunctionType instance)
    * bit_generator (ctypes.c_void_p)
    """

    is_error_ptr = cgutils.alloca_once_value(c.builder, cgutils.false_bit)
    extra_refs = []

    def clear_extra_refs():
        for _ref in extra_refs:
            c.pyapi.decref(_ref)

    def handle_failure():
        c.builder.store(cgutils.true_bit, is_error_ptr)
        clear_extra_refs()

    with ExitStack() as stack:

        def object_getattr_safely(obj, attr):
            attr_obj = c.pyapi.object_getattr_string(obj, attr)
            extra_refs.append(attr_obj)
            return attr_obj

        struct_ptr = cgutils.create_struct_proxy(typ)(c.context, c.builder)
        struct_ptr.parent = obj

        # Get the .ctypes attr
        ctypes_binding = object_getattr_safely(obj, 'ctypes')
        with cgutils.early_exit_if_null(c.builder, stack, ctypes_binding):
            handle_failure()

        # Look up the "state_address" member and wire it into the struct
        interface_state_address = object_getattr_safely(
            ctypes_binding, 'state_address')
        with cgutils.early_exit_if_null(
            c.builder, stack, interface_state_address
        ):
            handle_failure()

        setattr(struct_ptr, 'state_address',
                c.unbox(types.uintp, interface_state_address).value)

        # Look up the "state" member and wire it into the struct
        interface_state = object_getattr_safely(ctypes_binding, 'state')
        with cgutils.early_exit_if_null(c.builder, stack, interface_state):
            handle_failure()

        interface_state_value = object_getattr_safely(
            interface_state, 'value')
        with cgutils.early_exit_if_null(
            c.builder, stack, interface_state_value
        ):
            handle_failure()
        setattr(
            struct_ptr,
            'state',
            c.unbox(
                types.uintp,
                interface_state_value).value)

        # Want to store callable function pointers to these CFunctionTypes, so
        # import ctypes and use it to cast the CFunctionTypes to c_void_p and
        # store the results.
        # First find ctypes.cast, and ctypes.c_void_p
        ctypes_name = c.context.insert_const_string(c.builder.module, 'ctypes')
        ctypes_module = c.pyapi.import_module(ctypes_name)
        extra_refs.append(ctypes_module)
        with cgutils.early_exit_if_null(c.builder, stack, ctypes_module):
            handle_failure()

        ct_cast = object_getattr_safely(ctypes_module, 'cast')
        with cgutils.early_exit_if_null(c.builder, stack, ct_cast):
            handle_failure()

        ct_voidptr_ty = object_getattr_safely(ctypes_module, 'c_void_p')
        with cgutils.early_exit_if_null(c.builder, stack, ct_voidptr_ty):
            handle_failure()

        # This wires in the fnptrs referred to by name
        def wire_in_fnptrs(name):
            # Find the CFunctionType function
            interface_next_fn = c.pyapi.object_getattr_string(
                ctypes_binding, name)

            extra_refs.append(interface_next_fn)
            with cgutils.early_exit_if_null(
                c.builder, stack, interface_next_fn
            ):
                handle_failure()

            # Want to do ctypes.cast(CFunctionType, ctypes.c_void_p), create an
            # args tuple for that.
            args = c.pyapi.tuple_pack([interface_next_fn, ct_voidptr_ty])
            with cgutils.early_exit_if_null(c.builder, stack, args):
                handle_failure()
            extra_refs.append(args)

            # Call ctypes.cast()
            interface_next_fn_casted = c.pyapi.call(ct_cast, args)
            extra_refs.append(interface_next_fn_casted)

            # Fetch the .value attr on the resulting ctypes.c_void_p for storage
            # in the function pointer slot.
            interface_next_fn_casted_value = object_getattr_safely(
                interface_next_fn_casted, 'value')
            with cgutils.early_exit_if_null(
                c.builder, stack, interface_next_fn_casted_value
            ):
                handle_failure()

            # Wire up
            setattr(struct_ptr, f'fnptr_{name}',
                    c.unbox(types.uintp, interface_next_fn_casted_value).value)

        wire_in_fnptrs('next_double')
        wire_in_fnptrs('next_uint64')
        wire_in_fnptrs('next_uint32')

        clear_extra_refs()

    return NativeValue(
        struct_ptr._getvalue(), is_error=c.builder.load(is_error_ptr)
    )


_bit_gen_type = types.NumPyRandomBitGeneratorType('bit_generator')


@unbox(types.NumPyRandomGeneratorType)
def unbox_numpy_random_generator(typ, obj, c):
    """
    Here we're creating a NumPyRandomGeneratorType StructModel with
    following fields:
    * ('bit_generator', _bit_gen_type): The unboxed BitGenerator
    associated with this Generator object instance.
    * ('parent', types.pyobject): Pointer to the original Generator
    PyObject.
    * ('meminfo', types.MemInfoPointer(types.voidptr)): The information
    about the memory stored at the pointer (to the original Generator
    PyObject). This is useful for keeping track of reference counts
    within the Python runtime. Helps prevent cases where deletion happens
    in Python runtime without NRT being awareness of it.
    """
    is_error_ptr = cgutils.alloca_once_value(c.builder, cgutils.false_bit)

    with ExitStack() as stack:
        struct_ptr = cgutils.create_struct_proxy(typ)(c.context, c.builder)
        bit_gen_inst = c.pyapi.object_getattr_string(obj, 'bit_generator')
        with cgutils.early_exit_if_null(c.builder, stack, bit_gen_inst):
            c.builder.store(cgutils.true_bit, is_error_ptr)
        unboxed = c.unbox(_bit_gen_type, bit_gen_inst).value
        struct_ptr.bit_generator = unboxed
        struct_ptr.parent = obj
        NULL = cgutils.voidptr_t(None)
        struct_ptr.meminfo = c.pyapi.nrt_meminfo_new_from_pyobject(
            NULL,  # there's no data
            obj,   # the python object, the call to
            # nrt_meminfo_new_from_pyobject will py_incref
        )
        c.pyapi.decref(bit_gen_inst)

    return NativeValue(
        struct_ptr._getvalue(), is_error=c.builder.load(is_error_ptr)
    )


@box(types.NumPyRandomGeneratorType)
def box_numpy_random_generator(typ, val, c):
    inst = c.context.make_helper(c.builder, typ, val)
    obj = inst.parent
    res = cgutils.alloca_once_value(c.builder, obj)
    c.pyapi.incref(obj)
    # Steal NRT ref
    c.context.nrt.decref(c.builder, typ, val)
    return c.builder.load(res)
