"""
This file provides internal compiler utilities that support certain special
operations with numpy.
"""
import numpy as np

from numba import types
from numba.cgutils import unpack_tuple
from numba.extending import intrinsic
from numba.targets.imputils import impl_ret_new_ref



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
