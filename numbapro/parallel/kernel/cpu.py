import numpy
import numba
from multiprocessing import cpu_count
import threading
from Queue import Queue

import llvm.ee as _le
import llvm.core as _lc
import llvm.passes as _lp
from llvm.workaround.avx_support import detect_avx_support

from .cpuenv import CUEnvironment
from .cpudecor import cu_jit
from .cu import CU
from ._cpuscheduler import WorkGang

#
# CPU CU
#
class CPUComputeUnit(CU):
    def _init(self):
        from os.path import join, dirname
        self.__env = CUEnvironment.get_environment('numbapro.cu')
        # setup llvm engine
        with open(join(dirname(__file__), 'atomic.ll')) as fin:
            self.__module = _lc.Module.from_assembly(fin)
        self.__module.id = str(self)

        if not detect_avx_support():
            features = '-avx'
        else:
            features = ''
        tm = _le.TargetMachine.new(opt=2, cm=_le.CM_JITDEFAULT, features=features)
        passmanagers = _lp.build_pass_managers(tm, opt=2,
                                               inline_threshold=2000,
                                               loop_vectorize=True,
                                               fpm=False)
        self.__pm = passmanagers.pm
        self.__engine = _le.EngineBuilder.new(self.__module).create(tm)
        self.__engine.disable_lazy_compilation(True)

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
        self.__cpu_count = max(cpu_count() - 1, 1)
        assert self.__cpu_count > 0

    def _close(self):
        self.__queue.put(StopIteration)

    def __manager_logic(self, queue):
        from ctypes import addressof, sizeof
        while True:
            item = queue.get()
            if item is StopIteration:
                return
            entryptr, ntid, cargs = item

            gang = WorkGang(self.__cpu_count, entryptr, ntid,
                            addressof(cargs), sizeof(cargs),
                            self.__atomic_add_ptr)
            gang.join()
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
    # prepare
    module = kernel.module.clone()
    kernel = module.get_function_named(kernel.name)

    ity = _lc.Type.int()
    oargtys = kernel.type.pointee.args
    packedargs = _lc.Type.pointer(_lc.Type.struct(oargtys))
    argtys = [ity] * 2 + [packedargs]

    fnty = _lc.Type.function(_lc.Type.void(), argtys)
    func = module.add_function(fnty, 'wrapper.%s' % kernel.name)

    bbentry, bbloop, bbexit = (func.append_basic_block(x)
                               for x in ('entry', 'loop', 'exit'))

    const_int = lambda x: _lc.Constant.int(ity, x)

    # entry
    bldr = _lc.Builder.new(bbentry)
    begin, end, args = func.args
    
    innerargs = []
    for i in range(1, len(oargtys)):
        gep = bldr.gep(args, [const_int(0), const_int(i)])
        val = bldr.load(gep)
        innerargs.append(val)

    pred = bldr.icmp(_lc.ICMP_ULT, begin, end)
    bldr.cbranch(pred, bbloop, bbexit)

    # loop body
    bldr.position_at_end(bbloop)
    tid = bldr.phi(ity)
    tidnxt = bldr.add(tid, const_int(1))

    tid.add_incoming(begin, bbentry)
    tid.add_incoming(tidnxt, bbloop)

    # call
    bldr.call(kernel, [tid] + innerargs)

    pred = bldr.icmp(_lc.ICMP_ULT, tidnxt, end)
    bldr.cbranch(pred, bbloop, bbexit)

    # exit
    bldr.position_at_end(bbexit)
    bldr.ret_void()

    # verify
    failed = func.verify()
    assert not failed

    return module, func



CU.registered_targets['cpu'] = CPUComputeUnit

