"""
This scripts specifies all PTX special objects.
"""
import operator
import numpy
import llvmlite.llvmpy.core as lc
from numba.core import types, typing, ir
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
# syncthreads

class syncthreads(Stub):
    '''
    Synchronize all threads in the same thread block.  This function implements
    the same pattern as barriers in traditional multi-threaded programming: this
    function waits until all threads in the block call it, at which point it
    returns control to all its callers.
    '''
    _description_ = '<syncthreads()>'


class syncthreads_count(Stub):
    '''
    syncthreads_count(predictate)

    An extension to numba.cuda.syncthreads where the return value is a count
    of the threads where predicate is true.
    '''
    _description_ = '<syncthreads_count()>'


class syncthreads_and(Stub):
    '''
    syncthreads_and(predictate)

    An extension to numba.cuda.syncthreads where 1 is returned if predicate is
    true for all threads or 0 otherwise.
    '''
    _description_ = '<syncthreads_and()>'


class syncthreads_or(Stub):
    '''
    syncthreads_or(predictate)

    An extension to numba.cuda.syncthreads where 1 is returned if predicate is
    true for any thread or 0 otherwise.
    '''
    _description_ = '<syncthreads_or()>'


# -------------------------------------------------------------------------------
# warp level operations

class syncwarp(Stub):
    '''
    syncwarp(mask)

    Synchronizes a masked subset of threads in a warp.
    '''
    _description_ = '<warp_sync()>'


class shfl_sync_intrinsic(Stub):
    '''
    shfl_sync_intrinsic(mask, mode, value, mode_offset, clamp)

    Nvvm intrinsic for shuffling data across a warp
    docs.nvidia.com/cuda/nvvm-ir-spec/index.html#nvvm-intrin-warp-level-datamove
    '''
    _description_ = '<shfl_sync()>'


class vote_sync_intrinsic(Stub):
    '''
    vote_sync_intrinsic(mask, mode, predictate)

    Nvvm intrinsic for performing a reduce and broadcast across a warp
    docs.nvidia.com/cuda/nvvm-ir-spec/index.html#nvvm-intrin-warp-level-vote
    '''
    _description_ = '<vote_sync()>'


class match_any_sync(Stub):
    '''
    match_any_sync(mask, value)

    Nvvm intrinsic for performing a compare and broadcast across a warp.
    Returns a mask of threads that have same value as the given value from within the masked warp.
    '''
    _description_ = '<match_any_sync()>'


class match_all_sync(Stub):
    '''
    match_all_sync(mask, value)

    Nvvm intrinsic for performing a compare and broadcast across a warp.
    Returns a tuple of (mask, pred), where mask is a mask of threads that have
    same value as the given value from within the masked warp, if they
    all have the same value, otherwise it is 0. Pred is a boolean of whether
    or not all threads in the mask warp have the same warp.
    '''
    _description_ = '<match_all_sync()>'


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


#-------------------------------------------------------------------------------
# bit manipulation

class popc(Stub):
    """
    popc(val)

    Returns the number of set bits in the given value.
    """


class brev(Stub):
    """
    brev(val)

    Reverse the bitpattern of an integer value; for example 0b10110110
    becomes 0b01101101.
    """


class clz(Stub):
    """
    clz(val)

    Counts the number of leading zeros in a value.
    """


class ffs(Stub):
    """
    ffs(val)

    Find the position of the least significant bit set to 1 in an integer.
    """

#-------------------------------------------------------------------------------
# comparison and selection instructions

class selp(Stub):
    """
    selp(a, b, c)

    Select between source operands, based on the value of the predicate source operand.
    """

#-------------------------------------------------------------------------------
# single / double precision arithmetic

class fma(Stub):
    """
    fma(a, b, c)

    Perform the fused multiply-add operation.
    """

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

        Returns the old value at the index location as if it is loaded
        atomically.
        """

    class max(Stub):
        """max(ary, idx, val)

        Perform atomic ary[idx] = max(ary[idx], val). NaN is treated as a
        missing value, so max(NaN, n) == max(n, NaN) == n. Note that this
        differs from Python and Numpy behaviour, where max(a, b) is always
        a when either a or b is a NaN.

        Supported on int32, int64, uint32, uint64, float32, float64 operands only.

        Returns the old value at the index location as if it is loaded
        atomically.
        """

    class min(Stub):
        """min(ary, idx, val)

        Perform atomic ary[idx] = min(ary[idx], val). NaN is treated as a
        missing value, so min(NaN, n) == min(n, NaN) == n. Note that this
        differs from Python and Numpy behaviour, where min(a, b) is always
        a when either a or b is a NaN.

        Supported on int32, int64, uint32, uint64, float32, float64 operands only.
        """

    class compare_and_swap(Stub):
        """compare_and_swap(ary, old, val)

        Conditionally assign ``val`` to the first element of an 1D array ``ary``
        if the current value matches ``old``.

        Returns the current value as if it is loaded atomically.
        """
