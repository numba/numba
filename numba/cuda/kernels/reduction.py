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

    # Ensure that there is enough shared memory for smaller block sizes
    sdatasize = max(blocksize, 64)

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
        sdata = cuda.shared.array(sdatasize, dtype=typ)

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
    # The reduction kernels are precompiled for all block sizes at the time when
    # the reduction operation is first called. A single reduction can use
    # multiple kernels specialized for different sizes. These compiled kernels
    # are cached inside the ``Reduce`` instance that created them.
    #
    # Keeping the instance alive can avoid re-compiling.

    def __init__(self, binop):
        """Create a reduction object that reduces values using a given binary
        function. The binary function is compiled once and cached inside this
        object. Keeping this object alive will prevent re-compilation.

        :param binop: A function to be compiled as a CUDA device function that
                      will be used as the binary operation for reduction on a
                      CUDA device. Internally, it is compiled using
                      ``cuda.jit(signature, device=True)``.
        """
        self._kernels = {}
        self._cached_types = set()
        self._binop = binop

    def _get_kernel(self, size, nbtype):
        blocksize = min(size // 2, 512)
        key = nbtype, blocksize
        return self._kernels[key], blocksize * 2

    def _type_and_size(self, dary, size):
        nbtype = from_dtype(dary.dtype)

        if size is None:
            # Use the array size if the `size` is not defined
            size = dary.size

        if size > dary.size:
            raise ValueError("size > array.size")

        return nbtype, size

    def _execute_reduction(self, dary, size, stream, init=0):

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
        while size >= 2:
            # Find the closest size that is power of two, and the remainder
            p2size = 2 ** int(math.log(size, 2))
            diffsize = size - p2size

            # Generate a kernel to reduce the first p2size elements
            kernel, blocksize = self._get_kernel(p2size, nbtype)
            size = p2size
            gridsize = size // blocksize
            assert gridsize <= p2size
            if gridsize > 0:
                worksize = blocksize * gridsize
                blocksize = blocksize // 2
                assert size - worksize == 0
                # The reduction kernel stores its result in the first element of
                # its assigned sub-vector
                kernel[gridsize, blocksize, stream](dary, dary, worksize,
                                                    blocksize)
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

        # Finalise reduction with initial value if necessary
        if init != 0:
            kernel, _ = self._get_kernel(0, nbtype)
            kernel(dary, init)

    def _precompile_kernels(self, nbtype):
        from numba import cuda

        # Compile the reduction template for all required block sizes. See the
        # section "Invoking Template Kernels"
        if nbtype not in self._cached_types:
            for blocksize in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512):
                key = nbtype, blocksize
                self._kernels[key] = reduction_template(self._binop, nbtype,
                                                        blocksize)

            # Precompile the kernel for reducing with the initial value. We use
            # a block size of 0 as a placeholder for the reduction with the
            # initial value.
            binop = cuda.jit((nbtype, nbtype), device=True)(self._binop)

            @cuda.jit((nbtype[:], nbtype))
            def reduction_finish(arr, init):
                arr[0] = binop(arr[0], init)

            self._kernels[nbtype, 0] = reduction_finish

            self._cached_types.add(nbtype)

    def __call__(self, arr, res=None, size=None, init=0, stream=0):
        """Performs a full reduction.

        :param arr: A host or device array. If a device array is given, the
                    reduction is performed inplace and the values in the array
                    are overwritten. If a host array is given, it is copied to
                    the device automatically.
        :param size: Optional integer specifying the number of elements in
                     ``arr`` to reduce. If this parameter is not specified, the
                     entire array is reduced.
        :param res: Optional device array into which to write the reduction
                    result to. The result is written into the first element of
                    this array. If this parameter is specified, then no
                    communication of the reduction output takes place from the
                    device to the host.
        :param init: Optional initial value for the reduction, the type of which
                     must match ``arr.dtype``.
        :param stream: Optional CUDA stream in which to perform the reduction.
                       If no stream is specified, the default stream of 0 is
                       used.
        :return: If ``res`` is specified, ``None`` is returned. Otherwise, the
                 result of the reduction is returned.
        """
        from numba import cuda

        if arr.ndim != 1:
            raise TypeError("only support 1D array")

        # Avoid computation for zero-length input
        if (size is not None and size == 0) or (arr.size == 0):
            if res is None:
                return init
            else:
                res[0] = init
                return None

        darr, _ = cuda.devicearray.auto_device(arr, stream=stream)
        nbtype = from_dtype(arr.dtype)
        self._precompile_kernels(nbtype)
        self._execute_reduction(darr, size, stream, init=init)

        if res is None:
            return darr[0]
        else:
            view = darr.bind(stream=stream)[:1]
            res.copy_to_device(view, stream=stream)
            return None
