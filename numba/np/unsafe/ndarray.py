"""
This file provides internal compiler utilities that support certain special
operations with numpy.
"""
<<<<<<< HEAD:numba/np/unsafe/ndarray.py
from numba.core import types, typing
from numba.core.cgutils import unpack_tuple
from numba.core.extending import intrinsic
from numba.core.imputils import impl_ret_new_ref
from numba.core.errors import RequireLiteralValue, TypingError
=======
from numba import types
from numba import typing
from numba.cgutils import unpack_tuple
from numba.extending import intrinsic
from numba.targets.imputils import impl_ret_new_ref
from numba.targets.tupleobj import make_tuple
from numba.errors import RequireLiteralValue, TypingError
>>>>>>> 4b2be7389 (squashed dependencies (PR #5169 and #5173)):numba/unsafe/ndarray.py

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
    # QUESTION: Should this intrinsic be removed, gracefully deprecated or left
    # as is?
    if not isinstance(length, types.IntegerLiteral):
        raise RequireLiteralValue('*length* argument must be a constant')

    if array.ndim != 1:
        raise TypingError("Not supported on array.ndim={}".format(array.ndim))

    # Determine types
    tuple_size = int(length.literal_value)
    tuple_type = types.UniTuple(dtype=array.dtype, count=tuple_size)
    sig = tuple_type(array, length)

    def codegen(context, builder, signature, args):
        def impl(array):
            return make_tuple(tuple_size, array)

        inner_argtypes = [signature.args[0]]
        inner_sig = typing.signature(tuple_type, *inner_argtypes)
        inner_args = [args[0]]
        res = context.compile_internal(builder, impl, inner_sig, inner_args)
        return res

    return sig, codegen

