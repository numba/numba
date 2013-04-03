import math
import numpy
import numba
from weakref import WeakValueDictionary
from numbapro import cuda
from numba.decorators import resolve_argtypes
from numbapro.cudapipeline.environment import CudaEnvironment
from numbapro.cudapipeline.devicearray import DeviceArray
from numbapro.cudapipeline.error import CudaDriverError
from ..cu import CU


class WeakSet(object):
    def __init__(self):
        self._con = WeakValueDictionary()

    def add(self, obj):
        self._con[id(obj)] = obj

    def remove(self, obj):
        del self._con[id(obj)]

    def __iter__(self):
        return iter(v for k, v in self._con.iteritems())

    def __contains__(self, obj):
        return id(obj) in self._con

#
# CUDA CU
#

class CUDAComputeUnit(CU):
    def _init(self):
        self.__compiled_count = 0
        self.__env = CudaEnvironment.get_environment('numbapro.cuda')
        self.__stream = cuda.stream()
        self.__wrcache = WeakSet()
        self.__kernel_cache = {}
        self.__enqueued_args = []
        self.__device = cuda.get_current_device()
        self.__warpsize = self.__device.WARP_SIZE
        self.__max_blocksize = self.__device.MAX_THREADS_PER_BLOCK
        if self.__device.COMPUTE_CAPABILITY <= 2:
            # defined in CUDA-C Programming Guide
            self.__max_resident_block = 8
        else:
            self.__max_resident_block = 16

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

        self.__try_to_launch_kernel(jittedkern, ntid, args)
        self.__enqueued_args.extend(args) # keep ref to args

    def __try_to_launch_kernel(self, kernel, ntid, args):
        '''Try to lauch the CUDA kernel and
        try to optimize grid and block dimension to maximize occupancy.'''
        blksz = self.__max_blocksize
        WARPSZ = self.__warpsize
        while True:
            # optimize to maximum # of resident blocks
            nthread = min(blksz, ntid)
            nblock = int(math.ceil(float(ntid) / nthread))
            if nblock < self.__max_resident_block and nblock > WARPSZ:
                blksz -= WARPSZ
            else:
                break
        while True:
            griddim = nblock, 1,
            blockdim = nthread, 1
            # run kernel
            try:
                kernel[griddim, blockdim, self.__stream](ntid, *args)
            except CudaDriverError, e:
                # assuming the reason is too much registers
                # but who knows when the driver is not specific
                # so the user has to suffer for the delay
                blksz -= WARPSZ # reduce 32 threads at a time
                nthread = min(blksz, ntid)
                if nthread <= 0:
                    raise Exception("Cannot launch kernel for unknown reason.")
                nblock = int(math.ceil(float(ntid) / nthread))
            else:
                break

    def _run_epilog(self):
        # process device->host memory transfer
        uids = set()
        for arg in self.__enqueued_args:
            argid = id(arg) # use id because arg may not be hashable
            if argid not in uids: # new
                uids.add(argid)
                if self._isoutarray(arg):
                    arg.to_host(stream=self.__stream)

    def _wait(self):
        self._run_epilog()
        self.__stream.synchronize()
        self.__enqueued_args = [] # reset

    def _isoutarray(self, obj):
        try:
            return obj in self.__wrcache
        except:
            raise

    def _input(self, ary):
        return cuda.to_device(ary, stream=self.__stream)

    def _output(self, ary):
        devary = cuda.to_device(ary, copy=False, stream=self.__stream)
        self.__wrcache.add(devary)
        return devary

    def _inout(self, ary):
        devary = self._input(ary)
        self.__wrcache.add(devary)
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

