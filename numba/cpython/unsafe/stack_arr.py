#Workaround using stack-allocated arrays,
#faster than heap-allocated arrays.
#Workaround for ctypes byref
from numba import types,typeof,carray,farray,njit
from numba.extending import intrinsic
from numba.core import cgutils,errors


@intrinsic
def val_to_ptr(typingctx, data):
    """
    Get the value, pointed to by a pointer
    """
    def impl(context, builder, signature, args):
        ptr = cgutils.alloca_once_value(builder,args[0])
        return ptr
    sig = types.CPointer(typeof(data).instance_type)(typeof(
        data).instance_type)
    return sig, impl


@intrinsic
def ptr_to_val(typingctx, data):
    """
    Get a pointer to a given value
    """
    def impl(context, builder, signature, args):
        val = builder.load(args[0])
        return val
    sig = data.dtype(types.CPointer(data.dtype))
    return sig, impl


@intrinsic
def intr_stack_empty_alloc(typingctx,shape,dtype):
    size = 1
    if isinstance(shape,types.scalars.IntegerLiteral):
        size = shape.literal_value
        sig = types.CPointer(dtype.dtype)(types.int64,dtype)

    elif isinstance(shape,(types.containers.Tuple,
                           types.containers.UniTuple)):
        for i in range(len(shape)):
            size *= shape[i].literal_value

        sig = types.CPointer(dtype.dtype)(typeof(shape).instance_type,dtype)
    else:
        raise errors.TypingError(
            "Shape must be IntegerLiteral " +
            "or a tuple of IntegerLiterals")

    def impl(context, builder, signature, args):
        ty = context.get_value_type(dtype.dtype)
        ptr = cgutils.alloca_once(builder, ty,size=size)
        return ptr
    return sig, impl


#inline always, stack memory can't be returned
@njit(inline="always")
def stack_empty(shape,dtype,order="C"):
    """
    Allocate an array on the stack.
    Please note: - Arrays allocated on the stack can't be returned.
                 - Stack size is limited
    """
    arr_ptr = intr_stack_empty_alloc(shape,dtype)
    if order == "C":
        return carray(arr_ptr,shape)
    elif order == "F":
        return farray(arr_ptr,shape)
    else:
        raise errors.UnsupportedError(
            "order must be one of 'C', 'F'")
