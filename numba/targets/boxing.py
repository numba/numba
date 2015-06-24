"""
Boxing and unboxing of native Numba values to / from CPython objects.
"""

from llvmlite import ir

from .. import cgutils, numpy_support, types
from ..pythonapi import box


#
# Scalar types
#

@box(types.Boolean)
def box_bool(c, typ, val):
    longval = c.builder.zext(val, c.pyapi.long)
    return c.pyapi.bool_from_long(longval)

@box(types.Integer)
def box_integer(c, typ, val):
    if typ.signed:
        ival = c.builder.sext(val, c.pyapi.longlong)
        return c.pyapi.long_from_longlong(ival)
    else:
        ullval = c.builder.zext(val, c.pyapi.ulonglong)
        return c.pyapi.long_from_ulonglong(ullval)

@box(types.Float)
def box_float(c, typ, val):
    if typ == types.float32:
        dbval = c.builder.fpext(val, c.pyapi.double)
    else:
        assert typ == types.float64
        dbval = val
    return c.pyapi.float_from_double(dbval)

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

@box(types.NoneType)
def box_none(c, typ, val):
    return c.pyapi.make_none()

@box(types.NPDatetime)
def box_npdatetime(c, typ, val):
    return c.pyapi.create_np_datetime(val, typ.unit_code)

@box(types.NPTimedelta)
def box_nptimedelta(c, typ, val):
    return c.pyapi.create_np_timedelta(val, typ.unit_code)

@box(types.Record)
def box_record(c, typ, val):
    # Note we will create a copy of the record
    # This is the only safe way.
    size = ir.Constant(ir.IntType(32), val.type.pointee.count)
    ptr = c.builder.bitcast(val, ir.PointerType(ir.IntType(8)))
    return c.pyapi.recreate_record(ptr, size, typ.dtype, c.env_manager)

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

@box(types.Array)
def box_array(c, typ, val):
    nativearycls = c.context.make_array(typ)
    nativeary = nativearycls(c.context, c.builder, value=val)
    if c.context.enable_nrt:
        np_dtype = numpy_support.as_dtype(typ.dtype)
        dtypeptr = c.env_manager.read_const(c.env_manager.add_const(np_dtype))
        newary = c.pyapi.nrt_adapt_ndarray_to_python(typ, val, dtypeptr)
        return newary
    else:
        parent = nativeary.parent
        c.pyapi.incref(parent)
        return parent

@box(types.BaseTuple)
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


#
# Other types
#

@box(types.Generator)
def box_generator(c, typ, val):
    return c.pyapi.from_native_generator(val, typ)

@box(types.DType)
def box_dtype(c, typ, val):
    np_dtype = numpy_support.as_dtype(typ.dtype)
    return c.pyapi.unserialize(c.pyapi.serialize_object(np_dtype))

@box(types.PyObject)
def box_pyobject(c, typ, val):
    return val
