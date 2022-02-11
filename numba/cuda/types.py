from numba.core import types, cgutils
from numba.core.errors import NumbaNotImplementedError
from numba.np import numpy_support
from numba.core.pythonapi import box, unbox, NativeValue


class Dim3(types.Type):
    """
    A 3-tuple (x, y, z) representing the position of a block or thread.
    """
    def __init__(self):
        super().__init__(name='Dim3')


class GridGroup(types.Type):
    """
    The grid of all threads in a cooperative kernel launch.
    """
    def __init__(self):
        super().__init__(name='GridGroup')


dim3 = Dim3()
grid_group = GridGroup()


class CUDADispatcher(types.Dispatcher):
    """The type of CUDA dispatchers"""
    # This type exists (instead of using types.Dispatcher as the type of CUDA
    # dispatchers) so that we can have an alternative lowering for them to the
    # lowering of CPU dispatchers - the CPU target lowers all dispatchers as a
    # constant address, but we need to lower to a dummy value because it's not
    # generally valid to use the address of CUDA kernels and functions.
    #
    # Notes: it may be a bug in the CPU target that it lowers all dispatchers to
    # a constant address - it should perhaps only lower dispatchers acting as
    # first-class functions to a constant address. Even if that bug is fixed, it
    # is still probably a good idea to have a separate type for CUDA
    # dispatchers, and this type might get other differentiation from the CPU
    # dispatcher type in future.


class CUDADeviceArray(types.Array):
    """Type of a CUDA device array"""
    def __init__(self, *args, **kwargs):
        super(CUDADeviceArray, self).__init__(*args, **kwargs)
        self.name = f"CUDADevice{self.name}"


@unbox(CUDADeviceArray)
def unbox_cda(typ, obj, c):
    struct_ptr = cgutils.create_struct_proxy(typ)(c.context, c.builder)

    #('meminfo', types.MemInfoPointer(fe_type.dtype)),
    #('parent', types.pyobject),
    #('nitems', types.intp),
    #('itemsize', types.intp),
    #('data', types.CPointer(fe_type.dtype)),
    #('shape', types.UniTuple(types.intp, ndim)),
    #('strides', types.UniTuple(types.intp, ndim)),

    ndim = getattr(typ, 'ndim')
    attr_map = {"nitems": types.intp,
                "itemsize": types.intp,
                "shape": types.UniTuple(types.intp, ndim),
                "strides": types.UniTuple(types.intp, ndim),}

    # Just rewire this stuff
    for da_member, nb_type in attr_map.items():
        ref = c.pyapi.object_getattr_string(obj, da_member)
        setattr(struct_ptr, da_member, c.unbox(nb_type, ref).value)

    # Need to stuff in the native alloc ptr for "data"
    # d_x.gpu_data._mem.device_pointer_value

    # Get gpu_data
    gpu_data_owned_ptr = c.pyapi.object_getattr_string(obj, "gpu_data")

    # get mem *AutoFreePointer*
    afp = c.pyapi.object_getattr_string(gpu_data_owned_ptr, "_mem")
    # this is a unsigned long value of the address of the data
    dptr = c.pyapi.object_getattr_string(afp, "device_pointer_value")
    # unbox it
    unboxed_addr_native = c.unbox(types.uintp, dptr)
    failed = unboxed_addr_native.is_error
    with c.builder.if_then(failed, likely=False):
        c.pyapi.err_set_string("PyExc_TypeError",
                               "problem 1")
    # get value
    unboxed_addr = unboxed_addr_native.value

    # int2ptr device alloc addr then put into struct
    ll_uintp_ptr = c.context.get_value_type(types.uintp).as_pointer()
    unboxed_addr_as_ptr = c.builder.inttoptr(unboxed_addr, ll_uintp_ptr)
    casted = c.builder.bitcast(unboxed_addr_as_ptr, struct_ptr.data.type)
    setattr(struct_ptr, "data", casted)

    # meminfo
    null_meminfo = c.context.get_constant_null(types.MemInfoPointer(typ.dtype))
    setattr(struct_ptr, "meminfo", null_meminfo)

    # parent
    null_parent = c.context.get_constant_null(types.pyobject)
    setattr(struct_ptr, "parent", null_parent)

    return NativeValue(struct_ptr._getvalue())


@box(CUDADeviceArray)
def box_array(typ, val, c):
    struct_ptr = c.context.make_helper(c.builder, typ, val)

    attr_map = {"nitems": types.intp,
                "itemsize": types.intp,
                "shape": types.UniTuple(types.intp, typ.ndim),
                "strides": types.UniTuple(types.intp, typ.ndim),}

    nitems_obj = c.box(attr_map['nitems'], struct_ptr.nitems)
    c.pyapi.incref(nitems_obj)
    shape_obj = c.box(attr_map['shape'], struct_ptr.shape)
    c.pyapi.incref(shape_obj)
    strides_obj = c.box(attr_map['strides'], struct_ptr.strides)
    c.pyapi.incref(strides_obj)
    np_dtype = numpy_support.as_dtype(typ.dtype)
    dtype_obj= c.env_manager.read_const(c.env_manager.add_const(np_dtype))
    ll_uintp = c.context.get_value_type(types.uintp)
    # HACK: needs a fix, stream not always 0
    stream_obj= c.env_manager.read_const(c.env_manager.add_const(0))
    c.pyapi.incref(stream_obj)

    # undo cast
    ll_uintp_ptr = c.context.get_value_type(types.uintp).as_pointer()
    casted = c.builder.bitcast(struct_ptr.data, ll_uintp_ptr)

    # undo inttoptr
    ll_uintp = c.context.get_value_type(types.uintp)
    boxed_addr = c.builder.ptrtoint(casted, ll_uintp)

    # box
    data_obj = c.box(types.intp, boxed_addr)
    c.pyapi.incref(data_obj)

    # NOTE: The data_obj is an int, should it really be a
    # MemoryPointer/OwnedPointer class?

    #def __init__(self, shape, strides, dtype, stream=0, gpu_data=None):
    from numba.cuda.cudadrv.devicearray import DeviceNDArrayBase
    cls_obj = c.pyapi.unserialize(c.pyapi.serialize_object(DeviceNDArrayBase))
    c.pyapi.incref(cls_obj)
    return c.pyapi.call_function_objargs(cls_obj, (shape_obj, strides_obj, dtype_obj, stream_obj, data_obj))

