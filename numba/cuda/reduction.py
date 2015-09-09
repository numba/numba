"""
A library written in CUDA Python for generating reduction kernels
"""
from __future__ import print_function, division, absolute_import
from functools import reduce
import math
from numba.numpy_support import from_dtype
from numba.types import uint32, intp

# This implementations is based on patterns described in "Optimizing Parallel
# Reduction In CUDA" by Mark Harris:
# http://developer.download.nvidia.com/compute/cuda/1.1-Beta/x86_website/projects/reduction/doc/reduction.pdf
# - sections of this document are referred to throughout this code.
#
# In this implementation, a tree-based approach to performing the reduction in
# parallel is taken, with threads in one block co-operating to reduce a
# sub-vector into a single value. Each block's partial reduction result is
# written out to produce a vector of partially-reduced results. Reduction
# kernels are invoked repeatedly until the remaining vector of partially-reduced
# results contains 16 or fewer elements, at which point the vector of partial
# results is transferred to the CPU for a final reduction to a single value.
#
# NOTE: This transfer of to the CPU for the final reduction is not an optimal
# strategy, as it required a roundtrip transferring data between the CPU and GPU
# when the reduction forms part of a set of operations on data on the GPU.
# Remedying this by performing the final reduction on the GPU is a TODO item -
# see https://github.com/numba/numba/issues/1407

def reduction_template(binop, typ, blocksize):
    """
    Returns a kernel that performs a partial reduction. This implementation is
    similar to that shown in the section "Reduction #6" of "Optimizing Parallel
    Reduction in CUDA" by Mark Harris.

    Args
    ----
    binop : function object
        A binary function as the reduction operation
    typ : numba type
        The numba type to the reduction operation
    blocksize : int
        The CUDA block size (thread per block)
    """
    from numba import cuda

    if blocksize > 512:
        # The reducer implementation is limited to 512 threads per block. This
        # is not an issue in practice, since the reduction function does not
        # instantiate this template with a blocksize greater than 512.
        raise ValueError("blocksize too big")

    # Compile binary operation as device function
    binop = cuda.jit((typ, typ), device=True)(binop)

    # Compile reducer
    @cuda.jit((typ[:], typ[:], intp, intp))
    def reducer(inp, out, nelem, ostride):
        tid = cuda.threadIdx.x
        i = cuda.blockIdx.x * (blocksize * 2) + tid
        gridSize = blocksize * 2 * cuda.gridDim.x

        # Blocks perform most of the reduction within shared memory, in the
        # sdata array
        sdata = cuda.shared.array(blocksize, dtype=typ)

        # The first reduction operation is performed during the process of
        # loading the data from global memory, in order to reduce the number of
        # idle threads (See "Reduction #4: First Add During Load")
        while i < nelem:
            sdata[tid] = binop(inp[i], inp[i + blocksize])
            i += gridSize

        # The following reduction steps rely on all values being loaded into
        # sdata; we need to synchronize in order to meet this condition
        cuda.syncthreads()

        # The following lines implement an unrolled loop that repeatedly reduces
        # the number of values by two (by performing the reduction operation)
        # until only a single value is left. This is done to reduce instruction
        # overhead (See the section "Instruction Bottleneck")
        if blocksize >= 512:
            if tid < 256:
                sdata[tid] = binop(sdata[tid], sdata[tid + 256])
                cuda.syncthreads()

        if blocksize >= 256:
            if tid < 128:
                sdata[tid] = binop(sdata[tid], sdata[tid + 128])
                cuda.syncthreads()

        if blocksize >= 128:
            if tid < 64:
                sdata[tid] = binop(sdata[tid], sdata[tid + 64])
                cuda.syncthreads()

        # At this point only the first warp has any work to do - we perform a
        # check on the thread ID here so that we can avoid calling syncthreads
        # (operations are synchronous within a warp) and also to avoid checking
        # the thread ID at each iteration (See the section "Unrolling the Last
        # Warp)
        if tid < 32:
            if blocksize >= 64:
                sdata[tid] = binop(sdata[tid], sdata[tid + 32])
            if blocksize >= 32:
                sdata[tid] = binop(sdata[tid], sdata[tid + 16])
            if blocksize >= 16:
                sdata[tid] = binop(sdata[tid], sdata[tid + 8])
            if blocksize >= 8:
                sdata[tid] = binop(sdata[tid], sdata[tid + 4])
            if blocksize >= 4:
                sdata[tid] = binop(sdata[tid], sdata[tid + 2])
            if blocksize >= 2:
                sdata[tid] = binop(sdata[tid], sdata[tid + 1])

        # Write this block's partially reduced value into the vector of all
        # partially-reduced values.
        if tid == 0:
            out[cuda.blockIdx.x * ostride] = sdata[0]

    # Return reducer
    return reducer


class Reduce(object):
    # The reduction kernels are compiled lazily as needed. A single reduction
    # can use multiple kernels specialized for different sizes. These compiled
    # kernels are cached inside the ``Reduce`` instance that created them.
    #
    # Keeping the instance alive can avoid re-compiling.
    #
    # The reduction kernel may not fully reduce the array on device.
    # The last few elements (usually less than 16) is copied back to the host
    # for the final reduction.

    def __init__(self, binop):
        """Uses binop as the binary operation for reduction.
        Uses ``cuda.jit(signature, device=True)`` to compile.

        Args
        -----
        binop: function
            A function to be compiled as a CUDA device function to be used
            as the binary operation for reduction on a CUDA device.

        Notes
        -----
        Function are compiled once and cached inside this object.  Keep this
        object alive will prevent re-compilation.
        """
        self.binop = binop
        self._cache = {}

    def _prepare(self, arr, stream):
        if arr.ndim != 1:
            raise TypeError("only support 1D array")

        from numba import cuda
        # If no stream is specified, allocate one
        if stream == 0:
            stream = cuda.stream()

        # Make sure `arr` in on the device
        darr, conv = cuda.devicearray.auto_device(arr, stream=stream)

        return darr, stream, conv

    def _type_and_size(self, dary, size):
        nbtype = from_dtype(dary.dtype)

        if size is None:
            # Use the array size if the `size` is not defined
            size = dary.size

        if size > dary.size:
            raise ValueError("size > array.size")

        return nbtype, size

    def device_partial_inplace(self, darr, size=None, init=0, stream=0):
        """Partially reduce a device array inplace as much as possible in an
        efficient manner. Does not automatically transfer host array.

        :param darr: Used to input and output.
        :type darr: device array
        :param size: Number of element in ``arr``.  If None, the entire array is used.
        :type size: int or None
        :param init: Initial value for the reduction
        :type init: dtype of darr
        :param stream: All CUDA operations are performed on this stream if it is given.
          Otherwise, a new stream is created.
        :type stream: cuda stream
        :returns: int -- Number of elements in ``darr`` that contains the reduction result.

        """
        if stream == 0:
            from numba import cuda
            stream = cuda.stream()
            ret = self._partial_inplace_driver(darr, size, init, stream)
            stream.synchronize()
        else:
            ret = self._partial_inplace_driver(darr, size, init, stream)
        return ret

    def _partial_inplace_driver(self, dary, size, init, stream):

        from numba import cuda

        nbtype, size = self._type_and_size(dary, size)

        # Partial reduction kernels write their output strided - this kernel is
        # used to copy the strided output to a compacted array which can be used
        # as input to the next kernel.
        @cuda.jit
        def copy_strides(arr, n, stride, tpb):
            sm = cuda.shared.array(1, dtype=uint32)
            i = cuda.threadIdx.x
            base = 0
            if i == 0:
                sm[0] = 0

            val = arr[0]
            while base < n:
                idx = base + i
                if idx < n:
                    val = arr[idx * stride]

                cuda.syncthreads()

                if base + i < n:
                    arr[sm[0] + i] = val

                if i == 0:
                    sm[0] += tpb

                base += tpb

        # The reduction is split into multiple steps, because each block writes
        # a partial result out, which cannot be used as input to other blocks
        # within the same kernel launch because this would require a global
        # synchronization. The global synchronization takes place between kernel
        # launches. See the section "Problem: Global Synchronization".
        while size >= 16:
            # Find the closest size that is power of two, and the remainder
            p2size = 2 ** int(math.log(size, 2))
            diffsize = size - p2size

            # Generate a kernel to reduce the first p2size elements
            kernel, blocksize = self._instantiate_template(p2size, nbtype, init)

            size = p2size
            gridsize = size // blocksize
            assert gridsize <= p2size
            if gridsize > 0:
                worksize = blocksize * gridsize
                blocksize = blocksize // 2
                assert size - worksize == 0
                # The reduction kernel stores its result in the first element of
                # its assigned sub-vector
                kernel[gridsize, blocksize, stream](dary, dary, worksize, blocksize)
                # Make the partial results contiguous by copying them to the
                # beginning of the vector
                copy_strides[1, 512, stream](dary, gridsize, blocksize, 512)
                # New size is gridsize, because there are exactly as many
                # elements remaining as there were thread blocks.
                size = gridsize

            # Compact any leftover elements beyond the p2size-th element of the
            # input vector by appending them to the contiguous vector of partial
            # results from the reduction kernel.
            if diffsize > 0:
                itemsize = dary.dtype.itemsize
                dst = dary.gpu_data.view(size * itemsize)
                src = dary.gpu_data.view(p2size * itemsize)
                cuda.driver.device_to_device(dst, src, diffsize * itemsize,
                                             stream=stream)
                size += diffsize

        return size

    def __call__(self, arr, size=None, init=0, stream=0):
        """Performs a full reduction.

        Returns the result of the full reduction

        Args
        ----
        arr : host or device array
            If a device array is given, the reduction is performed inplace.
            The values in the array may be overwritten.
            If a host array is given, it is copied to the device automatically.

        size : int or None
            Number of element in ``arr``.  If None, the entire array is used.

        init : dtype of darr
            Initial value for the reduction

        stream : cuda stream
            All CUDA operations are performed on this stream if it is given.
            Otherwise, a new stream is created.

        Returns
        -------
        depends on ``arr.dtype``
            Reduction result.

        Notes
        -----
        Calls ``device_partial_inplace`` internally.
        """
        if (size is not None and size == 0) or (arr.size == 0):
            return init
        darr, stream, conv = self._prepare(arr, stream)
        size = self._partial_inplace_driver(darr, size=size, init=init,
                                            stream=stream)
        hary = darr.bind(stream=stream)[:size].copy_to_host(stream=stream)
        return reduce(self.binop, hary, init)

    def _instantiate_template(self, size, nbtype, init):
        """Compile the kernel necessary for a reduction operation on a vector
        of a given size.
        """
        # The number and blocksize of kernels used in the operation depends on
        # the length of the vector - for longer vectors, a larger blocksize is
        # needed as there is enough work for many threads. For shorter vectors,
        # smaller blocksizes are more efficient.
        if size >= 1024:
            return self._compile(nbtype, 512, init)
        elif size >= 128:
            return self._compile(nbtype, 64, init)
        elif size >= 16:
            return self._compile(nbtype, 8, init)
        else:
            # When there are fewer than 16 elements, the reduction is not
            # performed on the GPU.
            raise ValueError('size < 16 unsupported')

    def _compile(self, nbtype, blockSize, init):
        """Compile a kernel for the parameter.

        Compiled kernels are cached.
        """
        key = nbtype, blockSize, init
        reducer = self._cache.get(key)
        if reducer is None:
            reducer = reduction_template(self.binop, nbtype, blockSize)
            self._cache[key] = reducer
        return reducer, blockSize * 2
