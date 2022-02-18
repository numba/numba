"""Implementation of Stack-allocated arrays
Please note that stack memory is limited and can't be returned.
There is no further check! If you try to return a stack array and
the function isn't inlined, you get an array object which points to
unallocated memory.

Purpose:
  - Small temporary arrays which are not retuned (faster allocation)
  - Small temporary arrays needed to wrap a C/Fortran function
"""
from numba import types,typeof,carray,farray,njit
from numba.extending import intrinsic
from numba.core import cgutils,errors
from numba.core.typing.npydecl import parse_dtype


@intrinsic
def val_to_ptr(typingctx, data):
    """
    Get a pointer of type typeof(value) to a value
    Python scalars are immuteable. Internally a stack-array is allocated
    and the given value is copied into it. To get a Python scalar ptr_to_val
    has to be called.
    This pointer is only valid within a function and can't be returned.
    """
    def impl(context, builder, signature, args):
        ptr = cgutils.alloca_once_value(builder, args[0])
        return ptr
    sig = types.CPointer(typeof(data).instance_type)(typeof(
        data).instance_type)
    return sig, impl


@intrinsic
def ptr_to_val(typingctx, data):
    """
    Get the first value, where a pointer points to
    """
    def impl(context, builder, signature, args):
        val = builder.load(args[0])
        return val
    sig = data.dtype(types.CPointer(data.dtype))
    return sig, impl


@intrinsic
def intr_stack_empty_alloc(typingctx, shape, dtype):
    """
    Allocates memory on the stack
    """
    size = 1
    val_type = parse_dtype(dtype)

    if isinstance(shape, types.scalars.IntegerLiteral):
        size = shape.literal_value
        sig = types.CPointer(val_type)(types.int64,dtype)

    elif isinstance(shape,(types.containers.Tuple)):
        for i in range(len(shape)):
            if isinstance(shape[i], types.scalars.IntegerLiteral):
                size *= shape[i].literal_value
            else:
                raise errors.LiteralTypingError(
                    "Shape " + str(i) + " is not a Integer Literal")

        sig = types.CPointer(val_type)(typeof(shape).instance_type,dtype)
    else:
        raise errors.TypingError(
            "Shape must be IntegerLiteral " +
            "or a tuple of IntegerLiterals")

    def impl(context, builder, signature, args):
        ty = context.get_value_type(val_type)
        ptr = cgutils.alloca_once(builder, ty,size=size)
        return ptr
    return sig, impl


#inline always, stack memory can't be returned
@njit(inline="always")
def empty(shape, dtype, order="C"):
    """
    Allocate an empty array on the stack.
    Please note: - Arrays allocated on the stack can't be returned.
                 - Stack size is limited
    """
    arr_ptr = intr_stack_empty_alloc(shape, dtype)
    if order == "C":
        return carray(arr_ptr, shape)
    elif order == "F":
        return farray(arr_ptr, shape)
    else:
        raise errors.UnsupportedError(
            "order must be one of 'C', 'F'")


#inline always, stack memory can't be returned
@njit(inline="always")
def zeros(shape, dtype, order="C"):
    """
    Allocate an array filled with zeros on the stack.
    Please note: - Arrays allocated on the stack can't be returned.
                 - Stack size is limited
    """
    arr_ptr = intr_stack_empty_alloc(shape, dtype)
    if order == "C":
        arr = carray(arr_ptr, shape)
        arr[:] = 0
        return arr
    elif order == "F":
        arr = farray(arr_ptr, shape)
        arr[:] = 0
        return arr
    else:
        raise errors.UnsupportedError(
            "order must be one of 'C', 'F'")
