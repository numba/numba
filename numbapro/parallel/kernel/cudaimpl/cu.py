import math
import numpy
import numba
from numbapro import cuda
from numba.decorators import resolve_argtypes
from numbapro.cudapipeline.environment import CudaEnvironment
from numbapro.cudapipeline.devicearray import DeviceArray

from ..cu import CU

#
# CUDA CU
#

class CUDAComputeUnit(CU):
    def _init(self):
        self.__compiled_count = 0
        self.__env = CudaEnvironment.get_environment('numbapro.cuda')
        self.__stream = cuda.stream()
        self.__kernel_cache = {}
        self.__writeback = set()
        self.__enqueued_args = []

    @property
    def _stream(self):
        return self.__stream

    def __typemap(self, values):
        typemapper = self.__env.context.typemapper.from_python
        for val in values:
            if isinstance(val, DeviceArray):
                shape = tuple(1 for _ in val.shape)
                fakearray = numpy.empty(shape, dtype=val.dtype)
                yield typemapper(fakearray)
            else:
                yield typemapper(val)

    def _execute_builtin(self, name, impl, ntid, args):
        impl(self, ntid, args)

    def _execute_kernel(self, func, ntid, args):
        # Compile device function
        origargtypes = list(self.__typemap(args))
        argtypes = [numba.int32] + origargtypes   # prepend tid
        compiler = cuda.jit(argtypes=argtypes, device=True, inline=True,
                            env=self.__env)
        corefn = compiler(func)

        jittedkern = self.__kernel_cache.get(corefn.lfunc.name)
        if jittedkern is None:

            # Prepare kernel function
            uid = self.__compiled_count
            kwds = dict(uid  = uid,
                        args = ', '.join("arg%d" % i 
                                         for i, _ in enumerate(args)),
                        core = 'corefn')
            kernelsource = cuda_driver_kernel_template.format(**kwds)

            # compile it
            cxt = dict(corefn=corefn, cuda=cuda)
            exec kernelsource in cxt
            kernelname = "cu_kernel_cuda_%d" % uid
            kernel = cxt[kernelname]

            jittedkern = cuda.jit(argtypes=argtypes, env=self.__env)(kernel)
            # cache it
            self.__kernel_cache[corefn.lfunc.name] = jittedkern
            # keep stats
            self.__compiled_count += 1

        # configure kernel
        blksz = 1024
        ngrid = int(math.ceil(float(ntid) / blksz))
        nblock = min(blksz, ntid)
        griddim = ngrid, 1,
        blockdim = nblock, 1

        # run kernel
        jittedkern[griddim, blockdim, self.__stream](ntid, *args)
        self.__enqueued_args.append(args) # keep ref to args

    
    def _enqueue_write(self, ary):
        self.__writeback.remove(ary)
        ary.to_host(stream=self.__stream)

    def _run_epilog(self):
        for mem in self.__writeback:
            mem.to_host(stream=self.__stream)

    def _wait(self):
        self._run_epilog()
        self.__stream.synchronize()
        self.__enqueued_args = [] # reset

    def _input(self, ary):
        return cuda.to_device(ary, stream=self.__stream)

    def _output(self, ary):
        devary = cuda.to_device(ary, copy=False, stream=self.__stream)
        self.__writeback.add(devary)
        return devary

    def _inout(self, ary):
        devary = self._input(ary)
        self.__writeback.add(devary)
        return devary

    def _scratch(self, shape, dtype, order):
        return cuda.device_array(shape = shape,
                                 dtype = dtype,
                                 order = order,
                                 stream = self.__stream)


cuda_driver_kernel_template = '''
def cu_kernel_cuda_{uid}(ntid, {args}):
    tid = cuda.grid(1)
    if tid >= ntid:
        return
    {core}(tid, {args})
'''

CU.registered_targets['gpu'] = CUDAComputeUnit

