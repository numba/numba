import math
import numpy
import numba
from collections import namedtuple

from numbapro import cuda
from numba.decorators import resolve_argtypes
from numbapro.cudapipeline.environment import CudaEnvironment
from numbapro.cudapipeline.devicearray import DeviceArray
from .environment import CUEnvironment
from .decorators import cu_jit

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

    def scratch(self, shape, dtype=numpy.float, order='C'):
        return self._scratch(shape=shape, dtype=dtype, order=order)

    def scratch_like(self, ary):
        order = ''
        if ary.flags['C_CONTIGUOUS']:
            order = 'C'
        elif ary.flags['F_CONTIGUOUS']:
            order = 'F'
        return self.scratch(shape=ary.shape, dtype=ary.dtype, order=order)

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

#
# CPU CU
#

class CPUComputeUnit(ComputeUnit):
    def _init(self):
        from llvm.ee import EngineBuilder, TargetMachine, CM_JITDEFAULT
        from llvm.passes import build_pass_managers
        from llvm.core import Module
        from llvm.workaround.avx_support import detect_avx_support
        from os.path import join, dirname
        from ctypes import c_void_p, CDLL, c_int
        from multiprocessing import cpu_count
        import threading
        from Queue import Queue

        self.__env = CUEnvironment.get_environment('numbapro.cu')

        # setup llvm engine
        with open(join(dirname(__file__), 'atomic.ll')) as fin:
            self.__module = Module.from_assembly(fin)
        self.__module.id = str(self)

        if not detect_avx_support():
            features = '-avx'
        else:
            features = ''
        tm = TargetMachine.new(opt=2, cm=CM_JITDEFAULT, features=features)
        passmanagers = build_pass_managers(tm, opt=2,
                                           inline_threshold=2000,
                                           loop_vectorize=True,
                                           fpm=False)
        self.__pm = passmanagers.pm
        self.__engine = EngineBuilder.new(self.__module).create(tm)
        self.__engine.disable_lazy_compilation(True)

        # ctypes
        self.__dll = dll = CDLL(join(dirname(__file__), 'cpuscheduler.so'))

        self.__start_workers = start_workers = dll.start_workers
        start_workers.argtypes = [c_int, c_void_p, c_int, c_void_p, c_int,
                                  c_void_p]
        start_workers.restype = c_void_p
        
        self.__join_workers = join_workers = dll.join_workers
        join_workers.argtypes = [c_void_p]
        
        # prepare dispatcher
        atomics = 'atomic_add_i32'
        atomic_add_fn = self.__module.get_function_named( 'atomic_add_i32')
        atomic_add_ptr = self.__engine.get_pointer_to_function(atomic_add_fn)
        self.__atomic_add_ptr = atomic_add_ptr

        # manager thread
        self.__queue = Queue()
        self.__manager = threading.Thread(target=(self.__manager_logic),
                                          args=(self.__queue,))
        self.__manager.daemon = True
        self.__manager.start()

        # others
        self.__kernel_cache = {}
        self.__cpu_count = cpu_count() - 1

    def __del__(self):
        self.__queue.put(StopIteration)

    def __manager_logic(self, queue):
        from ctypes import byref, sizeof
        while True:
            item = queue.get()
            if item is StopIteration:
                return
            entryptr, ntid, cargs = item
            gang = self.__start_workers(self.__cpu_count, entryptr, ntid,
                                        byref(cargs), sizeof(cargs),
                                        self.__atomic_add_ptr)
            self.__join_workers(gang)
            queue.task_done()

    def _execute_kernel(self, func, ntid, args):
        typemapper = self.__env.context.typemapper.from_python
        argtypes = [numba.int32] + list(map(typemapper, args))
        # compile with CUDA pipeline here to bypass linkage
        compiler = cu_jit(argtypes=argtypes)
        lfunc = compiler(func)
        entryptr = self.__kernel_cache.get(lfunc)
        if not entryptr:
            # kernel not already defined, generate it
            entryptr = self.__generate_kernel(lfunc)
        # start workers
        cargs = self.__get_carg(argtypes, args)
        self.__queue.put((entryptr, ntid, cargs))

    def __generate_kernel(self, lfunc):
        # make
        module, wrapper = make_cpu_kernel_wrapper(lfunc)
        # verify
        wrapper.module.verify()
        # optimize
        self.__pm.run(wrapper.module)
        # link
        wrapper_name = wrapper.name
        self.__module.link_in(module, preserve=True)
        entry = self.__module.get_function_named(wrapper_name)
        entryptr = self.__engine.get_pointer_to_function(entry)
        # cache
        self.__kernel_cache[lfunc] = entryptr
        return entryptr

    def __get_carg(self, argtypes, args):
        from ctypes import Structure, py_object, c_int
        fields = []
        for ty in argtypes:
            if ty.is_array:
                fields.append(py_object)
            else:
                fields.append(ty.to_ctypes())
        structfields = [("arg%d" % i, fd) for i, fd in enumerate(fields)]
        class Args(Structure):
            _fields_ = structfields
        cargs = Args(0, *args)
        return cargs


    def _wait(self):
        self.__queue.join()

    def _input(self, ary):
        return ary

    def _output(self, ary):
        return ary

    def _inout(self, ary):
        return ary

    def _scratch(self, shape, dtype, order):
        return numpy.empty(shape, dtype=dtype, order=order)


def make_cpu_kernel_wrapper(kernel):
    from llvm.core import Type, Builder, Constant, ICMP_ULT
    # prepare
    module = kernel.module.clone()
    kernel = module.get_function_named(kernel.name)

    ity = Type.int()
    oargtys = kernel.type.pointee.args
    packedargs = Type.pointer(Type.struct(oargtys))
    argtys = [ity] * 2 + [packedargs]

    fnty = Type.function(Type.void(), argtys)
    func = module.add_function(fnty, 'wrapper.%s' % kernel.name)

    bbentry, bbloop, bbexit = (func.append_basic_block(x)
                               for x in ('entry', 'loop', 'exit'))

    const_int = lambda x: Constant.int(ity, x)

    # entry
    bldr = Builder.new(bbentry)
    begin, end, args = func.args
    
    innerargs = []
    for i in range(1, len(oargtys)):
        gep = bldr.gep(args, [const_int(0), const_int(i)])
        val = bldr.load(gep)
        innerargs.append(val)

    pred = bldr.icmp(ICMP_ULT, begin, end)
    bldr.cbranch(pred, bbloop, bbexit)

    # loop body
    bldr.position_at_end(bbloop)
    tid = bldr.phi(ity)
    tidnxt = bldr.add(tid, const_int(1))

    tid.add_incoming(begin, bbentry)
    tid.add_incoming(tidnxt, bbloop)

    # call
    bldr.call(kernel, [tid] + innerargs)

    pred = bldr.icmp(ICMP_ULT, tidnxt, end)
    bldr.cbranch(pred, bbloop, bbexit)

    # exit
    bldr.position_at_end(bbexit)
    bldr.ret_void()

    # verify
    failed = func.verify()
    assert not failed

    return module, func



ComputeUnit.registered_targets['cpu'] = CPUComputeUnit


#
# CUDA CU
#

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

ComputeUnit.registered_targets['gpu'] = CUDAComputeUnit


# Short hand for compute unit
CU = ComputeUnit

