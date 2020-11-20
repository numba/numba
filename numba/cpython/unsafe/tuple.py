"""
This file provides internal compiler utilities that support certain special
operations with tuple and workarounds for limitations enforced in userland.
"""

from numba.core import types, typing
from numba.core.cgutils import alloca_once
from numba.core.extending import intrinsic


@intrinsic
def tuple_setitem(typingctx, tup, idx, val):
    """Return a copy of the tuple with item at *idx* replaced with *val*.

    Operation: ``out = tup[:idx] + (val,) + tup[idx + 1:]

    **Warning**

    - No boundchecking.
    - The dtype of the tuple cannot be changed.
      *val* is always cast to the existing dtype of the tuple.
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


@intrinsic
def build_full_slice_tuple(tyctx, sz):
    """Creates a sz-tuple of full slices."""
    size = int(sz.literal_value)
    tuple_type = types.UniTuple(dtype=types.slice2_type, count=size)
    sig = tuple_type(sz)

    def codegen(context, builder, signature, args):
        def impl(length, empty_tuple):
            out = empty_tuple
            for i in range(length):
                out = tuple_setitem(out, i, slice(None, None))
            return out

        inner_argtypes = [types.intp, tuple_type]
        inner_sig = typing.signature(tuple_type, *inner_argtypes)
        ll_idx_type = context.get_value_type(types.intp)
        # Allocate an empty tuple
        empty_tuple = context.get_constant_undef(tuple_type)
        inner_args = [ll_idx_type(size), empty_tuple]

        res = context.compile_internal(builder, impl, inner_sig, inner_args)
        return res

    return sig, codegen
