"""
This scripts specifies all PTX special objects.
"""
from __future__ import print_function, absolute_import, division
import operator
import numpy
import llvmlite.llvmpy.core as lc
from numba import types, ir, typing, macro
from .cudadrv import nvvm


class Stub(object):
    '''A stub object to represent special objects which is meaningless
    outside the context of CUDA-python.
    '''
    _description_ = '<ptx special value>'
    __slots__ = () # don't allocate __dict__

    def __new__(cls):
        raise NotImplementedError("%s is not instantiable" % cls)

    def __repr__(self):
        return self._description_

#-------------------------------------------------------------------------------
# SREG

SREG_SIGNATURE = typing.signature(types.int32)


class threadIdx(Stub):
    '''
    The thread indices in the current thread block, accessed through the
    attributes ``x``, ``y``, and ``z``. Each index is an integer spanning the
    range from 0 inclusive to the corresponding value of the attribute in
    :attr:`numba.cuda.blockDim` exclusive.
    '''
    _description_ = '<threadIdx.{x,y,z}>'

    x = macro.Macro('tid.x', SREG_SIGNATURE)
    y = macro.Macro('tid.y', SREG_SIGNATURE)
    z = macro.Macro('tid.z', SREG_SIGNATURE)


class blockIdx(Stub):
    '''
    The block indices in the grid of thread blocks, accessed through the
    attributes ``x``, ``y``, and ``z``. Each index is an integer spanning the
    range from 0 inclusive to the corresponding value of the attribute in
    :attr:`numba.cuda.gridDim` exclusive.
    '''
    _description_ = '<blockIdx.{x,y,z}>'

    x = macro.Macro('ctaid.x', SREG_SIGNATURE)
    y = macro.Macro('ctaid.y', SREG_SIGNATURE)
    z = macro.Macro('ctaid.z', SREG_SIGNATURE)


class blockDim(Stub):
    '''
    The shape of a block of threads, as declared when instantiating the
    kernel.  This value is the same for all threads in a given kernel, even
    if they belong to different blocks (i.e. each block is "full").
    '''
    x = macro.Macro('ntid.x', SREG_SIGNATURE)
    y = macro.Macro('ntid.y', SREG_SIGNATURE)
    z = macro.Macro('ntid.z', SREG_SIGNATURE)


class gridDim(Stub):
    '''
    The shape of the grid of blocks, accressed through the attributes ``x``,
    ``y``, and ``z``.
    '''
    _description_ = '<gridDim.{x,y,z}>'
    x = macro.Macro('nctaid.x', SREG_SIGNATURE)
    y = macro.Macro('nctaid.y', SREG_SIGNATURE)
    z = macro.Macro('nctaid.z', SREG_SIGNATURE)

#-------------------------------------------------------------------------------
# Grid Macro

def _ptx_grid1d(): pass


def _ptx_grid2d(): pass


def grid_expand(ndim):
    """grid(ndim)

    Return the absolute position of the current thread in the entire
    grid of blocks.  *ndim* should correspond to the number of dimensions
    declared when instantiating the kernel.  If *ndim* is 1, a single integer
    is returned.  If *ndim* is 2 or 3, a tuple of the given number of
    integers is returned.

    Computation of the first integer is as follows::

        cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x

    and is similar for the other two indices, but using the ``y`` and ``z``
    attributes.
    """
    if ndim == 1:
        fname = "ptx.grid.1d"
        restype = types.int32
    elif ndim == 2:
        fname = "ptx.grid.2d"
        restype = types.UniTuple(types.int32, 2)
    elif ndim == 3:
        fname = "ptx.grid.3d"
        restype = types.UniTuple(types.int32, 3)
    else:
        raise ValueError('argument can only be 1, 2, 3')

    return ir.Intrinsic(fname, typing.signature(restype, types.intp),
                        args=[ndim])

grid = macro.Macro('ptx.grid', grid_expand, callable=True)

#-------------------------------------------------------------------------------
# Gridsize Macro

def gridsize_expand(ndim):
    """
    Return the absolute size (or shape) in threads of the entire grid of
    blocks. *ndim* should correspond to the number of dimensions declared when
    instantiating the kernel.

    Computation of the first integer is as follows::

        cuda.blockDim.x * cuda.gridDim.x

    and is similar for the other two indices, but using the ``y`` and ``z``
    attributes.
    """
    if ndim == 1:
        fname = "ptx.gridsize.1d"
        restype = types.int32
    elif ndim == 2:
        fname = "ptx.gridsize.2d"
        restype = types.UniTuple(types.int32, 2)
    elif ndim == 3:
        fname = "ptx.gridsize.3d"
        restype = types.UniTuple(types.int32, 3)
    else:
        raise ValueError('argument can only be 1, 2 or 3')

    return ir.Intrinsic(fname, typing.signature(restype, types.intp),
                        args=[ndim])


gridsize = macro.Macro('ptx.gridsize', gridsize_expand, callable=True)

#-------------------------------------------------------------------------------
# synthreads

class syncthreads(Stub):
    '''
    Synchronize all threads in the same thread block.  This function implements
    the same pattern as barriers in traditional multi-threaded programming: this
    function waits until all threads in the block call it, at which point it
    returns control to all its callers.
    '''
    _description_ = '<syncthread()>'


# -------------------------------------------------------------------------------
# memory fences

class threadfence_block(Stub):
    '''
    A memory fence at thread block level
    '''
    _description_ = '<threadfence_block()>'


class threadfence_system(Stub):
    '''
    A memory fence at system level: across devices
    '''
    _description_ = '<threadfence_system()>'


class threadfence(Stub):
    '''
    A memory fence at device level
    '''
    _description_ = '<threadfence()>'


# -------------------------------------------------------------------------------
# shared

def _legalize_shape(shape):
    if isinstance(shape, tuple):
        return shape
    elif isinstance(shape, int):
        return (shape,)
    else:
        raise TypeError("invalid type for shape; got {0}".format(type(shape)))


def shared_array(shape, dtype):
    shape = _legalize_shape(shape)
    ndim = len(shape)
    fname = "ptx.smem.alloc"
    restype = types.Array(dtype, ndim, 'C')
    sig = typing.signature(restype, types.UniTuple(types.intp, ndim), types.Any)
    return ir.Intrinsic(fname, sig, args=(shape, dtype))


class shared(Stub):
    """
    Shared memory namespace.
    """
    _description_ = '<shared>'

    array = macro.Macro('shared.array', shared_array, callable=True,
                        argnames=['shape', 'dtype'])
    '''
    Allocate a shared array of the given *shape* and *type*. *shape* is either
    an integer or a tuple of integers representing the array's dimensions.
    *type* is a :ref:`Numba type <numba-types>` of the elements needing to be
    stored in the array.

    The returned array-like object can be read and written to like any normal
    device array (e.g. through indexing).
    '''


#-------------------------------------------------------------------------------
# local array


def local_array(shape, dtype):
    shape = _legalize_shape(shape)
    ndim = len(shape)
    fname = "ptx.lmem.alloc"
    restype = types.Array(dtype, ndim, 'C')
    sig = typing.signature(restype, types.UniTuple(types.intp, ndim), types.Any)
    return ir.Intrinsic(fname, sig, args=(shape, dtype))


class local(Stub):
    '''
    Local memory namespace.
    '''
    _description_ = '<local>'

    array = macro.Macro('local.array', local_array, callable=True,
                        argnames=['shape', 'dtype'])
    '''
    Allocate a local array of the given *shape* and *type*. The array is private
    to the current thread, and resides in global memory. An array-like object is
    returned which can be read and written to like any standard array (e.g.
    through indexing).
    '''

#-------------------------------------------------------------------------------
# const array


def const_array_like(ndarray):
    fname = "ptx.cmem.arylike"

    from .descriptor import CUDATargetDesc
    aryty = CUDATargetDesc.typingctx.resolve_argument_type(ndarray)

    sig = typing.signature(aryty, aryty)
    return ir.Intrinsic(fname, sig, args=[ndarray])


class const(Stub):
    '''
    Constant memory namespace.
    '''
    _description_ = '<const>'

    array_like = macro.Macro('const.array_like', const_array_like,
                             callable=True, argnames=['ary'])
    '''
    Create a const array from *ary*. The resulting const array will have the
    same shape, type, and values as *ary*.
    '''

#-------------------------------------------------------------------------------
# atomic

class atomic(Stub):
    """Namespace for atomic operations
    """
    _description_ = '<atomic>'

    class add(Stub):
        """add(ary, idx, val)

        Perform atomic ary[idx] += val. Supported on int32, float32, and
        float64 operands only.
        """

    class max(Stub):
        """max(ary, idx, val)

        Perform atomic ary[idx] = max(ary[idx], val). NaN is treated as a
        missing value, so max(NaN, n) == max(n, NaN) == n. Note that this
        differs from Python and Numpy behaviour, where max(a, b) is always
        a when either a or b is a NaN.

        Supported on float64 operands only.
        """
