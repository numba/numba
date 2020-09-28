"""
This file provides internal compiler utilities that support certain special
operations with numpy.
"""
from numba import literal_unroll
from numba.core import types, typing
from numba.core.cgutils import unpack_tuple
from numba.core.extending import intrinsic
from numba.core.imputils import impl_ret_new_ref
from numba.core.errors import RequireLiteralValue, TypingError

from numba.cpython.unsafe.tuple import tuple_setitem


@intrinsic
def empty_inferred(typingctx, shape):
    """A version of numpy.empty whose dtype is inferred by the type system.

    Expects `shape` to be a int-tuple.

    There is special logic in the type-inferencer to handle the "refine"-ing
    of undefined dtype.
    """
    from numba.np.arrayobj import _empty_nd_impl

    def codegen(context, builder, signature, args):
        # check that the return type is now defined
        arrty = signature.return_type
        assert arrty.is_precise()
        shapes = unpack_tuple(builder, args[0])
        # redirect implementation to np.empty
        res = _empty_nd_impl(context, builder, arrty, shapes)
        return impl_ret_new_ref(context, builder, arrty, res._getvalue())

    # make function signature
    nd = len(shape)
    array_ty = types.Array(ndim=nd, layout='C', dtype=types.undefined)
    sig = array_ty(shape)
    return sig, codegen


@intrinsic
def to_fixed_tuple(typingctx, array, length):
    """Convert *array* into a tuple of *length*

    Returns ``UniTuple(array.dtype, length)``

    ** Warning **
    - No boundchecking.
      If *length* is longer than *array.size*, the behavior is undefined.
    """
    if not isinstance(length, types.IntegerLiteral):
        raise RequireLiteralValue('*length* argument must be a constant')

    if array.ndim != 1:
        raise TypingError("Not supported on array.ndim={}".format(array.ndim))

    # Determine types
    tuple_size = int(length.literal_value)
    tuple_type = types.UniTuple(dtype=array.dtype, count=tuple_size)
    sig = tuple_type(array, length)

    indices = tuple(range(tuple_size))

    def codegen(context, builder, signature, args):
        def impl(array, empty_tuple):
            out = empty_tuple
            if tuple_size == 0:
                return out
            for i in literal_unroll(indices):
                out = tuple_setitem(out, i, array[i])
            return out

        inner_argtypes = [signature.args[0], tuple_type]
        inner_sig = typing.signature(tuple_type, *inner_argtypes)
        # Allocate an empty tuple
        empty_tuple = context.get_constant_undef(tuple_type)
        inner_args = [args[0], empty_tuple]

        res = context.compile_internal(builder, impl, inner_sig, inner_args)
        return res

    return sig, codegen

