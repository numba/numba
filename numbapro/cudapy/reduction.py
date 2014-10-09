"""
A library written in CUDA Python for generating reduction kernels
"""
from functools import reduce
from numbapro import cuda
from numba.numpy_support import from_dtype


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
    from numbapro import intp

    if blocksize > 512:
        # The reducer implementation is limited to 512 threads per block
        raise ValueError("blocksize too big")

    # Compile binary operation as device function
    binop = cuda.jit((typ, typ), device=True)(binop)

    # Compile reducer
    @cuda.jit((typ[:], typ[:], intp))
    def reducer(inp, out, nelem):
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
            out[cuda.blockIdx.x] = sdata[0]

    # Return reducer
    return reducer


class Reduce(object):
    """CUDA Reduce kernel
    """

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

    def __call__(self, arr, size=None, init=0, stream=0):
        if arr.ndim != 1:
            raise TypeError("only support 1D array")

        # If no stream is specified, allocate one
        if stream == 0:
            stream = cuda.stream()

        # Make sure `arr` in on the device
        darr = (cuda.to_device(arr, stream=stream)
                if not cuda.devicearray.is_cuda_ndarray(arr)
                else arr)

        return self._driver(darr, size=size, init=init, stream=stream)

    def _schedule_reducer(self, init, kernel, dary, dout, size, blockSize,
                          stream):
        gridsz = size // blockSize
        if gridsz > 0:
            worksz = blockSize * gridsz
            diffsz = size - worksz
            blocksz = blockSize // 2

            # Create an event if we have any leftovers
            if diffsz:
                evt = cuda.event()

            # Allocate output array if we don't have one already
            if dout is None or dout.size < gridsz:
                dout = cuda.device_array(gridsz, dtype=dary.dtype,
                                         stream=stream)

            # Launch reduction kernel
            kernel[gridsz, blocksz, stream](dary, dout, worksz)

            if diffsz:
                assert diffsz > 0
                hostary = dary.bind(stream=stream)[worksz:size].copy_to_host(
                    stream=stream)
                evt.record(stream=stream)

                def finish_diff():
                    # D->H transfer ready?
                    if not evt.query():
                        # Force host synchronization
                        evt.synchronize()
                    return reduce(self.binop, hostary, init)

                tail = finish_diff
            else:
                tail = lambda: init

            return dout, gridsz, tail

    def _compile(self, nbtype, blockSize, init):
        key = nbtype, blockSize, init
        reducer = self._cache.get(key)
        if reducer is None:
            reducer = reduction_template(self.binop, nbtype, blockSize)
            self._cache[key] = reducer
        return reducer, blockSize * 2

    def _driver(self, dary, size, init, stream):
        """Driver for compiling, scheduling, launching reduction kernel
        """
        nbtype = from_dtype(dary.dtype)

        if size is None:
            # Use the array size if the `size` is not defined
            size = dary.size

        if size > dary.size:
            raise ValueError("size > array.size")

        # Compile function
        plan = []
        if size >= 1024:
            plan.append(self._compile(nbtype, 512, init))
        if size >= 128:
            plan.append(self._compile(nbtype, 64, init))
        if size >= 16:
            plan.append(self._compile(nbtype, 8, init))

        # Run
        todos = []
        dout = None
        for kernel, blockSize in plan:
            while size != 1:
                args = self._schedule_reducer(init, kernel, dary, dout, size,
                                              blockSize, stream)
                if args is None:
                    break
                dout, size, tail = args
                dary, dout = dout, dary

                todos.append(tail)

        # Pull all the remaining parts and do reduce all them on the host
        final = reduce(self.binop, [fn() for fn in todos], init)
        hary = dary.bind(stream=stream)[:size].copy_to_host(stream=stream)

        stream.synchronize()
        return self.binop(reduce(self.binop, hary, init), final)

