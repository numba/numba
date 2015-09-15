"""
A library written in CUDA Python for generating reduction kernels
"""
from __future__ import print_function, division, absolute_import
from functools import reduce
import math
from numba.numpy_support import from_dtype
from numba.types import uint32

def reduction_template(binop, typ, blocksize):
    """
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
    from numba import intp

    if blocksize > 512:
        # The reducer implementation is limited to 512 threads per block
        raise ValueError("blocksize too big")

    # Compile binary operation as device function
    binop = cuda.jit((typ, typ), device=True)(binop)

    # Compile reducer
    @cuda.jit((typ[:], typ[:], intp, intp))
    def reducer(inp, out, nelem, ostride):
        tid = cuda.threadIdx.x
        i = cuda.blockIdx.x * (blocksize * 2) + tid
        gridSize = blocksize * 2 * cuda.gridDim.x
        sdata = cuda.shared.array(blocksize, dtype=typ)

        while i < nelem:
            sdata[tid] = binop(inp[i], inp[i + blocksize])
            i += gridSize

        cuda.syncthreads()

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



        while size >= 16:
            # Find the closest size that is power of two
            p2size = 2 ** int(math.log(size, 2))
            # Plan for p2size
            plan = self._plan(p2size, nbtype, init)

            diffsz = size - p2size
            size = p2size
            # Run kernels
            kernel, blockSize = plan[0]

            gridsz = size // blockSize
            assert gridsz <= p2size
            if gridsz > 0:
                worksz = blockSize * gridsz
                blocksz = blockSize // 2
                assert size - worksz == 0
                # Launch reduction kernel
                # Process data inplace
                # Result stored at start of each threadblock
                kernel[gridsz, blocksz, stream](dary, dary, worksz, blockSize)
                # Stream compact
                # Move all results to the start
                copy_strides[1, 512, stream](dary, gridsz, blockSize, 512)
                # New size is gridsz
                size = gridsz

            # Compact any leftover
            if diffsz > 0:
                from numba import cuda
                itemsize = dary.dtype.itemsize
                dst = dary.gpu_data.view(size * itemsize)
                src = dary.gpu_data.view(p2size * itemsize)
                cuda.driver.device_to_device(dst, src, diffsz * itemsize,
                                             stream=stream)
                size += diffsz

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

    def _plan(self, size, nbtype, init):
        """Compile the kernels necessary for the job.

        Compiled kernels are cached.
        """
        plan = []
        if size >= 1024:
            plan.append(self._compile(nbtype, 512, init))
        if size >= 128:
            plan.append(self._compile(nbtype, 64, init))
        if size >= 16:
            plan.append(self._compile(nbtype, 8, init))

        return plan

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
