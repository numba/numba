"""
Boxing and unboxing of native Numba values to / from CPython objects.
"""

from llvmlite import ir

from .. import cgutils, numpy_support, types
from ..pythonapi import box, unbox, reflect, NativeValue

from . import listobj


#
# Scalar types
#

@box(types.Boolean)
def box_bool(c, typ, val):
    longval = c.builder.zext(val, c.pyapi.long)
    return c.pyapi.bool_from_long(longval)

@unbox(types.Boolean)
def unbox_boolean(c, typ, obj):
    istrue = c.pyapi.object_istrue(obj)
    zero = ir.Constant(istrue.type, 0)
    val = c.builder.icmp_signed('!=', istrue, zero)
    return NativeValue(val, is_error=c.pyapi.c_api_error())


@box(types.Integer)
def box_integer(c, typ, val):
    if typ.signed:
        ival = c.builder.sext(val, c.pyapi.longlong)
        return c.pyapi.long_from_longlong(ival)
    else:
        ullval = c.builder.zext(val, c.pyapi.ulonglong)
        return c.pyapi.long_from_ulonglong(ullval)

@unbox(types.Integer)
def unbox_integer(c, typ, obj):
    ll_type = c.context.get_argument_type(typ)
    val = cgutils.alloca_once(c.builder, ll_type)
    longobj = c.pyapi.number_long(obj)
    with c.pyapi.if_object_ok(longobj):
        if typ.signed:
            llval = c.pyapi.long_as_longlong(longobj)
        else:
            llval = c.pyapi.long_as_ulonglong(longobj)
        c.pyapi.decref(longobj)
        c.builder.store(c.builder.trunc(llval, ll_type), val)
    return NativeValue(c.builder.load(val),
                       is_error=c.pyapi.c_api_error())


@box(types.Float)
def box_float(c, typ, val):
    if typ == types.float32:
        dbval = c.builder.fpext(val, c.pyapi.double)
    else:
        assert typ == types.float64
        dbval = val
    return c.pyapi.float_from_double(dbval)

@unbox(types.Float)
def unbox_float(c, typ, obj):
    fobj = c.pyapi.number_float(obj)
    dbval = c.pyapi.float_as_double(fobj)
    c.pyapi.decref(fobj)
    if typ == types.float32:
        val = c.builder.fptrunc(dbval,
                                c.context.get_argument_type(typ))
    else:
        assert typ == types.float64
        val = dbval
    return NativeValue(val, is_error=c.pyapi.c_api_error())


@box(types.Complex)
def box_complex(c, typ, val):
    cmplxcls = c.context.make_complex(typ)
    cval = cmplxcls(c.context, c.builder, value=val)

    if typ == types.complex64:
        freal = c.builder.fpext(cval.real, c.pyapi.double)
        fimag = c.builder.fpext(cval.imag, c.pyapi.double)
    else:
        assert typ == types.complex128
        freal, fimag = cval.real, cval.imag
    return c.pyapi.complex_from_doubles(freal, fimag)

@unbox(types.Complex)
def unbox_complex(c, typ, obj):
    c128cls = c.context.make_complex(types.complex128)
    c128 = c128cls(c.context, c.builder)
    ok = c.pyapi.complex_adaptor(obj, c128._getpointer())
    failed = cgutils.is_false(c.builder, ok)

    with cgutils.if_unlikely(c.builder, failed):
        c.pyapi.err_set_string("PyExc_TypeError",
                               "conversion to %s failed" % (typ,))

    if typ == types.complex64:
        cplxcls = c.context.make_complex(typ)
        cplx = cplxcls(c.context, c.builder)
        cplx.real = c.context.cast(c.builder, c128.real,
                                   types.float64, types.float32)
        cplx.imag = c.context.cast(c.builder, c128.imag,
                                   types.float64, types.float32)
    else:
        assert typ == types.complex128
        cplx = c128
    return NativeValue(cplx._getvalue(), is_error=failed)


@box(types.NoneType)
def box_none(c, typ, val):
    return c.pyapi.make_none()

@unbox(types.NoneType)
@unbox(types.EllipsisType)
def unbox_none(c, typ, val):
    return NativeValue(c.context.get_dummy_value())


@box(types.NPDatetime)
def box_npdatetime(c, typ, val):
    return c.pyapi.create_np_datetime(val, typ.unit_code)

@unbox(types.NPDatetime)
def unbox_npdatetime(c, typ, obj):
    val = c.pyapi.extract_np_datetime(obj)
    return NativeValue(val, is_error=c.pyapi.c_api_error())


@box(types.NPTimedelta)
def box_nptimedelta(c, typ, val):
    return c.pyapi.create_np_timedelta(val, typ.unit_code)

@unbox(types.NPTimedelta)
def unbox_nptimedelta(c, typ, obj):
    val = c.pyapi.extract_np_timedelta(obj)
    return NativeValue(val, is_error=c.pyapi.c_api_error())


@box(types.RawPointer)
def box_raw_pointer(c, typ, val):
    """
    Convert a raw pointer to a Python int.
    """
    ll_intp = c.context.get_value_type(types.uintp)
    addr = c.builder.ptrtoint(val, ll_intp)
    return c.box(types.uintp, addr)


#
# Composite types
#

@box(types.Record)
def box_record(c, typ, val):
    # Note we will create a copy of the record
    # This is the only safe way.
    size = ir.Constant(ir.IntType(32), val.type.pointee.count)
    ptr = c.builder.bitcast(val, ir.PointerType(ir.IntType(8)))
    return c.pyapi.recreate_record(ptr, size, typ.dtype, c.env_manager)

@unbox(types.Record)
def unbox_record(c, typ, obj):
    buf = c.pyapi.alloca_buffer()
    ptr = c.pyapi.extract_record_data(obj, buf)
    is_error = cgutils.is_null(c.builder, ptr)

    ltyp = c.context.get_value_type(typ)
    val = c.builder.bitcast(ptr, ltyp)

    def cleanup():
        c.pyapi.release_buffer(buf)
    return NativeValue(val, cleanup=cleanup, is_error=is_error)


@box(types.CharSeq)
def box_charseq(c, typ, val):
    rawptr = cgutils.alloca_once_value(c.builder, value=val)
    strptr = c.builder.bitcast(rawptr, c.pyapi.cstring)
    fullsize = c.context.get_constant(types.intp, typ.count)
    zero = c.context.get_constant(types.intp, 0)
    count = cgutils.alloca_once_value(c.builder, zero)

    bbend = c.builder.append_basic_block("end.string.count")

    # Find the length of the string
    with cgutils.loop_nest(c.builder, [fullsize], fullsize.type) as [idx]:
        # Get char at idx
        ch = c.builder.load(c.builder.gep(strptr, [idx]))
        # Store the current index as count
        c.builder.store(idx, count)
        # Check if the char is a null-byte
        ch_is_null = cgutils.is_null(c.builder, ch)
        # If the char is a null-byte
        with c.builder.if_then(ch_is_null):
            # Jump to the end
            c.builder.branch(bbend)

    # This is reached if there is no null-byte in the string
    # Then, set count to the fullsize
    c.builder.store(fullsize, count)
    # Jump to the end
    c.builder.branch(bbend)

    c.builder.position_at_end(bbend)
    strlen = c.builder.load(count)
    return c.pyapi.bytes_from_string_and_size(strptr, strlen)

@unbox(types.CharSeq)
def unbox_charseq(c, typ, obj):
    lty = c.context.get_value_type(typ)
    ok, buffer, size = c.pyapi.string_as_string_and_size(obj)

    # If conversion is ok, copy the buffer to the output storage.
    with cgutils.if_likely(c.builder, ok):
        # Check if the returned string size fits in the charseq
        storage_size = ir.Constant(size.type, typ.count)
        size_fits = c.builder.icmp_unsigned("<=", size, storage_size)

        # Allow truncation of string
        size = c.builder.select(size_fits, size, storage_size)

        # Initialize output to zero bytes
        null_string = ir.Constant(lty, None)
        outspace  = cgutils.alloca_once_value(c.builder, null_string)

        # We don't need to set the NULL-terminator because the storage
        # is already zero-filled.
        cgutils.memcpy(c.builder,
                       c.builder.bitcast(outspace, buffer.type),
                       buffer, size)

    ret = c.builder.load(outspace)
    return NativeValue(ret, is_error=c.builder.not_(ok))


@unbox(types.Optional)
def unbox_optional(c, typ, obj):
    """
    Convert object *obj* to a native optional structure.
    """
    noneval = c.context.make_optional_none(c.builder, typ.type)
    is_not_none = c.builder.icmp_signed('!=', obj, c.pyapi.borrow_none())

    retptr = cgutils.alloca_once(c.builder, noneval.type)
    errptr = cgutils.alloca_once_value(c.builder, cgutils.false_bit)

    with c.builder.if_else(is_not_none) as (then, orelse):
        with then:
            native = c.unbox(typ.type, obj)
            just = c.context.make_optional_value(c.builder,
                                                 typ.type, native.value)
            c.builder.store(just, retptr)
            c.builder.store(native.is_error, errptr)

        with orelse:
            c.builder.store(noneval, retptr)

    if native.cleanup is not None:
        def cleanup():
            with c.builder.if_then(is_not_none):
                native.cleanup()
    else:
        cleanup = None

    ret = c.builder.load(retptr)
    return NativeValue(ret, is_error=c.builder.load(errptr),
                       cleanup=cleanup)


@unbox(types.Slice3Type)
def unbox_slice(c, typ, obj):
    """
    Convert object *obj* to a native slice structure.
    """
    from . import slicing
    ok, start, stop, step = \
        c.pyapi.slice_as_ints(obj, slicing.get_defaults(c.context))
    slice3 = slicing.Slice(c.context, c.builder)
    slice3.start = start
    slice3.stop = stop
    slice3.step = step
    return NativeValue(slice3._getvalue(), is_error=c.builder.not_(ok))


#
# Collections
#

# NOTE: boxing functions are supposed to steal any NRT references in
# the given native value.

@box(types.Array)
def box_array(c, typ, val):
    nativearycls = c.context.make_array(typ)
    nativeary = nativearycls(c.context, c.builder, value=val)
    if c.context.enable_nrt:
        np_dtype = numpy_support.as_dtype(typ.dtype)
        dtypeptr = c.env_manager.read_const(c.env_manager.add_const(np_dtype))
        # Steals NRT ref
        newary = c.pyapi.nrt_adapt_ndarray_to_python(typ, val, dtypeptr)
        return newary
    else:
        parent = nativeary.parent
        c.pyapi.incref(parent)
        return parent

@unbox(types.Buffer)
def unbox_buffer(c, typ, obj):
    """
    Convert a Py_buffer-providing object to a native array structure.
    """
    buf = c.pyapi.alloca_buffer()
    res = c.pyapi.get_buffer(obj, buf)
    is_error = cgutils.is_not_null(c.builder, res)

    nativearycls = c.context.make_array(typ)
    nativeary = nativearycls(c.context, c.builder)
    aryptr = nativeary._getpointer()

    with cgutils.if_likely(c.builder, c.builder.not_(is_error)):
        ptr = c.builder.bitcast(aryptr, c.pyapi.voidptr)
        if c.context.enable_nrt:
            c.pyapi.nrt_adapt_buffer_from_python(buf, ptr)
        else:
            c.pyapi.numba_buffer_adaptor(buf, ptr)

    def cleanup():
        c.pyapi.release_buffer(buf)

    return NativeValue(c.builder.load(aryptr), is_error=is_error,
                       cleanup=cleanup)

@unbox(types.Array)
def unbox_array(c, typ, obj):
    """
    Convert a Numpy array object to a native array structure.
    """
    # This is necessary because unbox_buffer() does not work on some
    # dtypes, e.g. datetime64 and timedelta64.
    # TODO check matching dtype.
    #      currently, mismatching dtype will still work and causes
    #      potential memory corruption
    nativearycls = c.context.make_array(typ)
    nativeary = nativearycls(c.context, c.builder)
    aryptr = nativeary._getpointer()

    ptr = c.builder.bitcast(aryptr, c.pyapi.voidptr)
    if c.context.enable_nrt:
        errcode = c.pyapi.nrt_adapt_ndarray_from_python(obj, ptr)
    else:
        errcode = c.pyapi.numba_array_adaptor(obj, ptr)
    failed = cgutils.is_not_null(c.builder, errcode)
    return NativeValue(c.builder.load(aryptr), is_error=failed)


@box(types.Tuple)
@box(types.UniTuple)
def box_tuple(c, typ, val):
    """
    Convert native array or structure *val* to a tuple object.
    """
    tuple_val = c.pyapi.tuple_new(typ.count)

    for i, dtype in enumerate(typ):
        item = c.builder.extract_value(val, i)
        obj = c.box(dtype, item)
        c.pyapi.tuple_setitem(tuple_val, i, obj)

    return tuple_val

@box(types.NamedTuple)
@box(types.NamedUniTuple)
def box_namedtuple(c, typ, val):
    """
    Convert native array or structure *val* to a namedtuple object.
    """
    cls_obj = c.pyapi.unserialize(c.pyapi.serialize_object(typ.instance_class))
    tuple_obj = box_tuple(c, typ, val)
    obj = c.pyapi.call(cls_obj, tuple_obj)
    c.pyapi.decref(cls_obj)
    c.pyapi.decref(tuple_obj)
    return obj


@unbox(types.BaseTuple)
def unbox_tuple(c, typ, obj):
    """
    Convert tuple *obj* to a native array (if homogenous) or structure.
    """
    n = len(typ)
    values = []
    cleanups = []
    is_error = cgutils.false_bit
    for i, eltype in enumerate(typ):
        elem = c.pyapi.tuple_getitem(obj, i)
        native = c.unbox(eltype, elem)
        values.append(native.value)
        is_error = c.builder.or_(is_error, native.is_error)
        if native.cleanup is not None:
            cleanups.append(native.cleanup)

    if cleanups:
        def cleanup():
            for func in reversed(cleanups):
                func()
    else:
        cleanup = None

    if isinstance(typ, types.UniTuple):
        value = cgutils.pack_array(c.builder, values)
    else:
        value = cgutils.make_anonymous_struct(c.builder, values)
    return NativeValue(value, is_error=is_error, cleanup=cleanup)


@box(types.List)
def box_list(c, typ, val):
    """
    Convert native list *val* to a list object.
    """
    list = listobj.ListInstance(c.context, c.builder, typ, val)
    obj = list.parent
    res = cgutils.alloca_once_value(c.builder, obj)
    with c.builder.if_else(cgutils.is_not_null(c.builder, obj)) as (has_parent, otherwise):
        with has_parent:
            # List is actually reflected => return the original object
            # (note not all list instances whose *type* is reflected are
            #  actually reflected; see numba.tests.test_lists for an example)
            c.pyapi.incref(obj)

        with otherwise:
            # Build a new Python list
            nitems = list.size
            obj = c.pyapi.list_new(nitems)
            with c.builder.if_then(cgutils.is_not_null(c.builder, obj),
                                   likely=True):
                with cgutils.for_range(c.builder, nitems) as loop:
                    item = list.getitem(loop.index)
                    itemobj = c.box(typ.dtype, item)
                    c.pyapi.list_setitem(obj, loop.index, itemobj)

            c.builder.store(obj, res)

    # Steal NRT ref
    c.context.nrt_decref(c.builder, typ, val)
    return c.builder.load(res)


@unbox(types.List)
def unbox_list(c, typ, obj):
    """
    Convert list *obj* to a native list.

    If list was previously unboxed, we reuse the existing native list
    to ensure consistency.
    """
    size = c.pyapi.list_size(obj)

    errorptr = cgutils.alloca_once_value(c.builder, cgutils.false_bit)
    listptr = cgutils.alloca_once(c.builder, c.context.get_value_type(typ))

    # Use pointer-stuffing hack to see if the list was previously unboxed,
    # if so, re-use the meminfo.
    ptr = c.pyapi.list_get_private_data(obj)

    with c.builder.if_else(cgutils.is_not_null(c.builder, ptr)) \
        as (has_meminfo, otherwise):

        with has_meminfo:
            # List was previously unboxed => reuse meminfo
            list = listobj.ListInstance.from_meminfo(c.context, c.builder, typ, ptr)
            list.size = size
            if typ.reflected:
                list.parent = obj
            c.builder.store(list.value, listptr)

        with otherwise:
            # Allocate a new native list
            ok, list = listobj.ListInstance.allocate_ex(c.context, c.builder, typ, size)
            with c.builder.if_else(ok, likely=True) as (if_ok, if_not_ok):
                with if_ok:
                    list.size = size
                    with cgutils.for_range(c.builder, size) as loop:
                        itemobj = c.pyapi.list_getitem(obj, loop.index)
                        # XXX we don't call native cleanup for each
                        # list element, since that would require keeping
                        # of which unboxings have been successful.
                        native = c.unbox(typ.dtype, itemobj)
                        with c.builder.if_then(native.is_error, likely=False):
                            c.builder.store(cgutils.true_bit, errorptr)
                        list.setitem(loop.index, native.value)
                    if typ.reflected:
                        list.parent = obj
                    # Stuff meminfo pointer into the Python object for
                    # later reuse.
                    c.pyapi.list_set_private_data(obj, list.meminfo)
                    c.builder.store(list.value, listptr)
                with if_not_ok:
                    c.builder.store(cgutils.true_bit, errorptr)

            # If an error occurred, drop the whole native list
            with c.builder.if_then(c.builder.load(errorptr)):
                c.context.nrt_decref(c.builder, typ, list.value)

    def cleanup():
        # Clean up the stuffed pointer, as the meminfo is now invalid.
        c.pyapi.list_reset_private_data(obj)

    return NativeValue(c.builder.load(listptr),
                       is_error=c.builder.load(errorptr),
                       cleanup=cleanup)


@reflect(types.List)
def reflect_list(c, typ, val):
    """
    Reflect the native list's contents into the Python object.
    """
    if not typ.reflected:
        return
    list = listobj.ListInstance(c.context, c.builder, typ, val)
    with c.builder.if_then(list.dirty, likely=False):
        obj = list.parent
        size = c.pyapi.list_size(obj)
        new_size = list.size
        diff = c.builder.sub(new_size, size)
        diff_gt_0 = c.builder.icmp_signed('>=', diff,
                                          ir.Constant(diff.type, 0))
        with c.builder.if_else(diff_gt_0) as (if_grow, if_shrink):
            # XXX no error checking below
            with if_grow:
                # First overwrite existing items
                with cgutils.for_range(c.builder, size) as loop:
                    item = list.getitem(loop.index)
                    itemobj = c.box(typ.dtype, item)
                    c.pyapi.list_setitem(obj, loop.index, itemobj)
                # Then add missing items
                with cgutils.for_range(c.builder, diff) as loop:
                    idx = c.builder.add(size, loop.index)
                    item = list.getitem(idx)
                    itemobj = c.box(typ.dtype, item)
                    c.pyapi.list_append(obj, itemobj)
                    c.pyapi.decref(itemobj)

            with if_shrink:
                # First delete list tail
                c.pyapi.list_setslice(obj, new_size, size, None)
                # Then overwrite remaining items
                with cgutils.for_range(c.builder, new_size) as loop:
                    item = list.getitem(loop.index)
                    itemobj = c.box(typ.dtype, item)
                    c.pyapi.list_setitem(obj, loop.index, itemobj)

        # Mark the list clean, in case it is reflected twice
        list.set_dirty(False)


#
# Other types
#

@box(types.Generator)
def box_generator(c, typ, val):
    return c.pyapi.from_native_generator(val, typ, c.env_manager.env_ptr)

@unbox(types.Generator)
def unbox_generator(c, typ, obj):
    return c.pyapi.to_native_generator(obj, typ)


@box(types.DType)
def box_dtype(c, typ, val):
    np_dtype = numpy_support.as_dtype(typ.dtype)
    return c.pyapi.unserialize(c.pyapi.serialize_object(np_dtype))


@box(types.PyObject)
@box(types.Object)
def box_pyobject(c, typ, val):
    return val

@unbox(types.PyObject)
@unbox(types.Object)
def unbox_pyobject(c, typ, obj):
    return NativeValue(obj)


@unbox(types.ExternalFunctionPointer)
def unbox_funcptr(c, typ, obj):
    if typ.get_pointer is None:
        raise NotImplementedError(typ)

    # Call get_pointer() on the object to get the raw pointer value
    ptrty = c.context.get_function_pointer_type(typ)
    ret = cgutils.alloca_once_value(c.builder,
                                    ir.Constant(ptrty, None),
                                    name='fnptr')
    ser = c.pyapi.serialize_object(typ.get_pointer)
    get_pointer = c.pyapi.unserialize(ser)
    with cgutils.if_likely(c.builder,
                           cgutils.is_not_null(c.builder, get_pointer)):
        intobj = c.pyapi.call_function_objargs(get_pointer, (obj,))
        c.pyapi.decref(get_pointer)
        with cgutils.if_likely(c.builder,
                               cgutils.is_not_null(c.builder, intobj)):
            ptr = c.pyapi.long_as_voidptr(intobj)
            c.pyapi.decref(intobj)
            c.builder.store(c.builder.bitcast(ptr, ptrty), ret)
    return NativeValue(c.builder.load(ret), is_error=c.pyapi.c_api_error())
