"""
This scripts specifies all PTX special objects.
"""
import functools


class Stub(object):
    '''
    A stub object to represent special objects that are meaningless
    outside the context of a CUDA kernel
    '''
    _description_ = '<ptx special value>'
    __slots__ = () # don't allocate __dict__

    def __new__(cls):
        raise NotImplementedError("%s is not instantiable" % cls)

    def __repr__(self):
        return self._description_


def stub_function(fn):
    '''
    A stub function to represent special functions that are meaningless
    outside the context of a CUDA kernel
    '''
    @functools.wraps(fn)
    def wrapped(*args, **kwargs):
        raise NotImplementedError("%s cannot be called from host code" % fn)
    return wrapped


#-------------------------------------------------------------------------------
# Thread and grid indices and dimensions


class Dim3(Stub):
    '''A triple, (x, y, z)'''
    _description_ = '<Dim3>'

    @property
    def x(self):
        pass

    @property
    def y(self):
        pass

    @property
    def z(self):
        pass


class threadIdx(Dim3):
    '''
    The thread indices in the current thread block. Each index is an integer
    spanning the range from 0 inclusive to the corresponding value of the
    attribute in :attr:`numba.cuda.blockDim` exclusive.
    '''
    _description_ = '<threadIdx.{x,y,z}>'


class blockIdx(Dim3):
    '''
    The block indices in the grid of thread blocks. Each index is an integer
    spanning the range from 0 inclusive to the corresponding value of the
    attribute in :attr:`numba.cuda.gridDim` exclusive.
    '''
    _description_ = '<blockIdx.{x,y,z}>'


class blockDim(Dim3):
    '''
    The shape of a block of threads, as declared when instantiating the kernel.
    This value is the same for all threads in a given kernel launch, even if
    they belong to different blocks (i.e. each block is "full").
    '''
    _description_ = '<blockDim.{x,y,z}>'


class gridDim(Dim3):
    '''
    The shape of the grid of blocks. This value is the same for all threads in
    a given kernel launch.
    '''
    _description_ = '<gridDim.{x,y,z}>'


class warpsize(Stub):
    '''
    The size of a warp. All architectures implemented to date have a warp size
    of 32.
    '''
    _description_ = '<warpsize>'


class laneid(Stub):
    '''
    This thread's lane within a warp. Ranges from 0 to
    :attr:`numba.cuda.warpsize` - 1.
    '''
    _description_ = '<laneid>'


class grid(Stub):
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
    _description_ = '<grid(ndim)>'


class gridsize(Stub):
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
    _description_ = '<gridsize(ndim)>'


#-------------------------------------------------------------------------------
# Array creation

class shared(Stub):
    '''
    Shared memory namespace
    '''
    _description_ = '<shared>'

    @stub_function
    def array(shape, dtype):
        '''
        Allocate a shared array of the given *shape* and *type*. *shape* is
        either an integer or a tuple of integers representing the array's
        dimensions.  *type* is a :ref:`Numba type <numba-types>` of the
        elements needing to be stored in the array.

        The returned array-like object can be read and written to like any
        normal device array (e.g. through indexing).
        '''


class local(Stub):
    '''
    Local memory namespace
    '''
    _description_ = '<local>'

    @stub_function
    def array(shape, dtype):
        '''
        Allocate a local array of the given *shape* and *type*. The array is
        private to the current thread, and resides in global memory. An
        array-like object is returned which can be read and written to like any
        standard array (e.g.  through indexing).
        '''


class const(Stub):
    '''
    Constant memory namespace
    '''

    @stub_function
    def array_like(ndarray):
        '''
        Create a const array from *ndarry*. The resulting const array will have
        the same shape, type, and values as *ndarray*.
        '''


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
    syncwarp(mask=0xFFFFFFFF)

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
    Returns a mask of threads that have same value as the given value from
    within the masked warp.
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

    Select between source operands, based on the value of the predicate source
    operand.
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

    class sub(Stub):
        """sub(ary, idx, val)

        Perform atomic ary[idx] -= val. Supported on int32, float32, and
        float64 operands only.

        Returns the old value at the index location as if it is loaded
        atomically.
        """

    class max(Stub):
        """max(ary, idx, val)

        Perform atomic ary[idx] = max(ary[idx], val).

        Supported on int32, int64, uint32, uint64, float32, float64 operands
        only.

        Returns the old value at the index location as if it is loaded
        atomically.
        """

    class min(Stub):
        """min(ary, idx, val)

        Perform atomic ary[idx] = min(ary[idx], val).

        Supported on int32, int64, uint32, uint64, float32, float64 operands
        only.

        Returns the old value at the index location as if it is loaded
        atomically.
        """

    class nanmax(Stub):
        """nanmax(ary, idx, val)

        Perform atomic ary[idx] = max(ary[idx], val).

        NOTE: NaN is treated as a missing value such that:
        nanmax(NaN, n) == n, nanmax(n, NaN) == n

        Supported on int32, int64, uint32, uint64, float32, float64 operands
        only.

        Returns the old value at the index location as if it is loaded
        atomically.
        """

    class nanmin(Stub):
        """nanmin(ary, idx, val)

        Perform atomic ary[idx] = min(ary[idx], val).

        NOTE: NaN is treated as a missing value, such that:
        nanmin(NaN, n) == n, nanmin(n, NaN) == n

        Supported on int32, int64, uint32, uint64, float32, float64 operands
        only.

        Returns the old value at the index location as if it is loaded
        atomically.
        """

    class compare_and_swap(Stub):
        """compare_and_swap(ary, old, val)

        Conditionally assign ``val`` to the first element of an 1D array ``ary``
        if the current value matches ``old``.

        Returns the current value as if it is loaded atomically.
        """
