from numba import types
from numba.core import cgutils
from numba.core.typing import signature
from numba.cuda import nvvmutils
from numba.cuda.extending import intrinsic


def _type_grid_function(ndim):
    val = ndim.literal_value
    if val == 1:
        restype = types.int32
    elif val in (2, 3):
        restype = types.UniTuple(types.int32, val)
    else:
        raise ValueError('argument can only be 1, 2, 3')

    return signature(restype, types.int32)


@intrinsic
def grid(typingctx, ndim):
    '''grid(ndim)

    Return the absolute position of the current thread in the entire grid of
    blocks.  *ndim* should correspond to the number of dimensions declared when
    instantiating the kernel. If *ndim* is 1, a single integer is returned.
    If *ndim* is 2 or 3, a tuple of the given number of integers is returned.

    Computation of the first integer is as follows::

        cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x

    and is similar for the other two indices, but using the ``y`` and ``z``
    attributes.
    '''

    if not isinstance(ndim, types.IntegerLiteral):
        return None

    sig = _type_grid_function(ndim)

    def codegen(context, builder, sig, args):
        restype = sig.return_type
        if restype == types.int32:
            return nvvmutils.get_global_id(builder, dim=1)
        elif isinstance(restype, types.UniTuple):
            ids = nvvmutils.get_global_id(builder, dim=restype.count)
            return cgutils.pack_array(builder, ids)

    return sig, codegen


@intrinsic
def gridsize(typingctx, ndim):
    '''gridsize(ndim)

    Return the absolute size (or shape) in threads of the entire grid of
    blocks. *ndim* should correspond to the number of dimensions declared when
    instantiating the kernel. If *ndim* is 1, a single integer is returned.
    If *ndim* is 2 or 3, a tuple of the given number of integers is returned.

    Computation of the first integer is as follows::

        cuda.blockDim.x * cuda.gridDim.x

    and is similar for the other two indices, but using the ``y`` and ``z``
    attributes.
    '''

    if not isinstance(ndim, types.IntegerLiteral):
        return None

    sig = _type_grid_function(ndim)

    def _nthreads_for_dim(builder, dim):
        ntid = nvvmutils.call_sreg(builder, f"ntid.{dim}")
        nctaid = nvvmutils.call_sreg(builder, f"nctaid.{dim}")
        return builder.mul(ntid, nctaid)

    def codegen(context, builder, sig, args):
        restype = sig.return_type
        nx = _nthreads_for_dim(builder, 'x')

        if restype == types.int32:
            return nx
        elif isinstance(restype, types.UniTuple):
            ny = _nthreads_for_dim(builder, 'y')

            if restype.count == 2:
                return cgutils.pack_array(builder, (nx, ny))
            elif restype.count == 3:
                nz = _nthreads_for_dim(builder, 'z')
                return cgutils.pack_array(builder, (nx, ny, nz))

    return sig, codegen
