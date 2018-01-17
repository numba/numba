"""
This file provides internal compiler utilities that support certain special
operations with numpy.
"""
from numba import types
from numba import typing
from numba.cgutils import unpack_tuple, alloca_once
from numba.extending import intrinsic
from numba.targets.imputils import impl_ret_new_ref
from numba.errors import RequireConstValue


@intrinsic
def empty_inferred(typingctx, shape):
    """A version of numpy.empty whose dtype is inferred by the type system.

    Expects `shape` to be a int-tuple.

    There is special logic in the type-inferencer to handle the "refine"-ing
    of undefined dtype.
    """
    from numba.targets.arrayobj import _empty_nd_impl

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
def tuple_setitem(typingctx, tup, idx, val):
    """Return a copy of the tuple with item at *idx* replaced with *val*.

    Operation: ``out = tup[:idx] + (val,) + tup[idx + 1:]

    **Warning**

    - No boundchecking.
    """
    def codegen(context, builder, signature, args):
        tup, idx, val = args
        stack = alloca_once(builder, tup.type)
        builder.store(tup, stack)
        # Unsafe load on unchecked bounds.  Poison value maybe returned.
        offptr = builder.gep(stack, [idx.type(0), idx], inbounds=True)
        builder.store(val, offptr)
        return builder.load(stack)

    sig = tup(tup, idx, tup.dtype)
    return sig, codegen


@intrinsic(support_literals=True)
def to_fixed_tuple(typingctx, array, length):
    """Convert *array* into a tuple of *length*

    ** Warning **
    - No boundchecking.
      If *length* is longer than *array.size*, the behavior is undefined.
    """
    if not isinstance(length, types.Const):
        raise RequireConstValue('*length* argument must be a constant')

    # Determine types
    tuple_size = int(length.value)
    tuple_type = types.UniTuple(dtype=array.dtype, count=tuple_size)
    sig = tuple_type(array, length)

    def codegen(context, builder, signature, args):
        def impl(array, length, empty_tuple):
            out = empty_tuple
            for i in range(length):
                out = tuple_setitem(out, i, array[i])
            return out

        inner_argtypes = [signature.args[0], types.intp, tuple_type]
        inner_sig = typing.signature(tuple_type, *inner_argtypes)
        ll_idx_type = context.get_value_type(types.intp)
        # Allocate an empty tuple
        empty_tuple = context.get_constant_undef(tuple_type)
        inner_args = [args[0], ll_idx_type(tuple_size), empty_tuple]

        res = context.compile_internal(builder, impl, inner_sig, inner_args)
        return res

    return sig, codegen

