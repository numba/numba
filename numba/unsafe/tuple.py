
from numba.cgutils import alloca_once
from numba.extending import intrinsic


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


