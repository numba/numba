import math
import numpy
import numba
from collections import namedtuple

from numbapro import cuda
from numba.decorators import resolve_argtypes
from numbapro.cudapipeline.environment import CudaEnvironment
from numbapro.cudapipeline.devicearray import DeviceArray

from .builtins._declaration import Declaration

Task = namedtuple('Task', ['func', 'ntid', 'args'])

class ComputeUnit(object):
    registered_targets = {}

    def __new__(self, target):
        cls = self.registered_targets[target]
        obj = object.__new__(cls)
        return obj

    def __init__(self, target):
        self.__target = target
        self.__state = State()
        self._init()

    @property
    def _state(self):
        return self.__state

    @property
    def target(self):
        return self.__target

    def enqueue(self, fn, ntid, args=()):
        if isinstance(fn, Declaration):
            name, impl = fn.get_implementation(self.target)
            self._execute_builtin(name, impl, ntid, args)
        else:
            self._execute_kernel(fn, ntid, args)

    def wait(self):
        self._wait()

    def input(self, ary):
        return self._input(ary)

    def output(self, ary):
        return self._output(ary)

    def inout(self, ary):
        return self._inout(ary)

    def scratch(self, shape, dtype=numpy.float, strides=None, order='C'):
        return self._scratch(shape=shape, dtype=dtype, strides=strides,
                             order=order)

    def scratch_like(self, ary):
        order = ''
        if ary.flags['C_CONTIGUOUS']:
            order = 'C'
        elif ary.flags['F_CONTIGUOUS']:
            order = 'F'
        return self.scratch(shape=ary.shape, strides=ary.strides,
                            dtype=ary.dtype, order=order)

    #
    # Methods to be overrided in subclass
    #

    def _init(self):
        '''Object initialization method.
        
        Default behavior does nothing.
        '''
        pass

    def _execute_kernel(self, func, ntid, ins, outs):
        raise NotImplementedError
    
    def _wait(self):
        pass

class State(object):
    __slots__ = '_store'
    def __init__(self):
        self._store = {}

    def get(self, name, default=None):
        return self._store.get(name, default)

    def set(self, name, value):
        setattr(self, name, value)

    def __getattr__(self, name):
        assert not name.startswith('_')
        return self._store_[name]

    def __setattr__(self, name, value):
        if name.startswith('_'):
            super(State, self).__setattr__(name, value)
        else:
            self._store[name] = value


class CUDAComputeUnit(ComputeUnit):
    def _init(self):
        self.__compiled_count = 0
        self.__env = CudaEnvironment.get_environment('numbapro.cuda')
        self.__stream = cuda.stream()
        self.__devmem_cache = {}
        self.__kernel_cache = {}
        self.__writeback = set()
    
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
                        args = ', '.join("arg%d" % i for i, _ in enumerate(args)),
                        core = 'corefn')
            kernelsource = cuda_driver_kernel_template.format(**kwds)

            # compile it
            cxt = dict(corefn=corefn, cuda=cuda)
            exec kernelsource in cxt
            kernelname = "cu_kernel_%d" % uid
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

    def _run_epilog(self):
        for mem in self.__writeback:
            mem.to_host(stream=self.__stream)

    def _wait(self):
        self._run_epilog()
        self.__stream.synchronize()

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

    def _scratch(self, shape, dtype, strides, order):
        return cuda.device_array(shape = shape,
                                 strides = strides,
                                 dtype = dtype,
                                 order = order,
                                 stream = self.__stream)


cuda_driver_kernel_template = '''
def cu_kernel_{uid}(ntid, {args}):
    tid = cuda.grid(1)
    if tid >= ntid:
        return
    {core}(tid, {args})
'''

ComputeUnit.registered_targets['gpu'] = CUDAComputeUnit


# Short hand for compute unit
CU = ComputeUnit

