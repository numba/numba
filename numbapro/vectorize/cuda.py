from llvm_cbuilder import *
from llvm_cbuilder import shortnames as C
from llvm.core import *
from llvm.passes import *
from llvm.ee import *
import numpy as np
from . import _common
from ._common import _llvm_ty_to_numpy

from numbapro.translate import Translate

class _CudaStagingCaller(CDefinition):
    def body(self, *args, **kwargs):
        worker = self.depends(self.WorkerDef)
        inputs = args[:-2]
        output, ct = args[-2:]

        # get current thread index
        tid_x = self.get_intrinsic(INTR_PTX_READ_TID_X, [])
        ntid_x = self.get_intrinsic(INTR_PTX_READ_NTID_X, [])
        ctaid_x = self.get_intrinsic(INTR_PTX_READ_CTAID_X, [])

        tid = self.var_copy(tid_x())
        blkdim = self.var_copy(ntid_x())
        blkid = self.var_copy(ctaid_x())

        i = tid + blkdim * blkid
        with self.ifelse( i >= ct ) as ifelse: # stop condition
            with ifelse.then():
                self.ret()

        res = worker(*map(lambda x: x[i], inputs))
        res.value.calling_convention = CC_PTX_DEVICE
        output[i].assign(res)

        self.ret()

    @classmethod
    def specialize(cls, worker, fntype):
        cls._name_ = ("_cukernel_%s" % worker).replace('.', '_2E_')

        args = cls._argtys_ = []
        inargs = cls.InArgs = []

        # input arguments
        for i, ty in enumerate(fntype.args):
            cur = ('in%d' % i, cls._pointer(ty))
            args.append(cur)
            inargs.append(cur)

        # output arguments
        cur = ('out', cls._pointer(fntype.return_type))
        args.append(cur)
        cls.OutArg = cur

        # extra arguments
        args.append(('ct', C.int))

        cls.WorkerDef = worker

    @classmethod
    def _pointer(cls, ty):
        return C.pointer(ty)

class CudaVectorize(_common.GenericVectorize):
    def __init__(self, func):
        super(CudaVectorize, self).__init__(func)
        self.module = Module.new("ptx_%s" % func.func_name)

    def add(self, *args, **kwargs):
        kwargs.update({'module': self.module})
        with _common.redirect_print(self.log):
            t = Translate(self.pyfunc, *args, **kwargs)
            t.translate()
            self.translates.append(t)

    def build_ufunc(self):
        # quick & dirty tryout
        # PyCuda should be optional
        from pycuda import driver as cudriver
        from pycuda.autoinit import device, context # use default
        from math import ceil

        lfunclist = self._get_lfunc_list()

        # setup optimizer for the staging caller
        fpm = FunctionPassManager.new(self.module)
        pmbldr = PassManagerBuilder.new()
        pmbldr.opt_level = 3
        pmbldr.populate(fpm)


        functype_list = []
        for lfunc in lfunclist: # generate caller for all function
            lfty = lfunc.type.pointee
            nptys = tuple(map(np.dtype, map(_llvm_ty_to_numpy, lfty.args)))

            lcaller = self._build_caller(lfunc)
            fpm.run(lcaller)    # run the optimizer

            # unicode problem?
            fname = lcaller.name
            if type(fname) is unicode:
                fname = fname.encode('utf-8')
            functype_list.append([fname,
                                 _llvm_ty_to_numpy(lfty.return_type),
                                 nptys])

        # force inlining & trim internal function
        pm = PassManager.new()
        pm.add(PASS_INLINE)
        pm.run(self.module)

        # generate ptx asm
        cc = 'compute_%d%d' % device.compute_capability() # select device cc
        if HAS_PTX:
            arch = 'ptx%d' % C.intp.width # select by host pointer size
        elif HAS_NVPTX:
            arch = {32: 'nvptx', 64: 'nvptx64'}[C.intp.width]
        else:
            raise Exception("llvmpy does not have PTX/NVPTX support")
        assert C.intp.width in [32, 64]
        ptxtm = TargetMachine.lookup(arch, cpu=cc, opt=3) # TODO: ptx64 option
        ptxasm = ptxtm.emit_assembly(self.module)
        # print(ptxasm)

        # prepare device
        ptxmodule = cudriver.module_from_buffer(ptxasm)

        devattr = device.get_attributes()
        MAX_THREAD = devattr[cudriver.device_attribute.MAX_THREADS_PER_BLOCK]
        # Take a safer approach for MAX_THREAD is case our kernel uses too
        # much resources.
        #
        # TODO: Add some intelligence to the way we choose our MAX_THREAD.
        #       Like optimize for the maximum wrap occupancy.
        MAX_THREAD /= 2

        MAX_BLOCK = devattr[cudriver.device_attribute.MAX_BLOCK_DIM_X]


        # get function
        kernel_list = [(ptxmodule.get_function(name), retty, argty)
                         for name, retty, argty in functype_list]

        def _ufunc_hack(*args):
            # determine type & kernel
            # FIXME: this is just a hack currently
            tys = tuple(map(lambda x: x.dtype, args))
            for kernel, retty, argtys in kernel_list:
                if argtys == tys:
                    break

            # prepare broadcasted arrays
            bcargs = np.broadcast_arrays(*args)
            N = bcargs[0].shape[0]

            retary = np.empty(N, dtype=retty)

            # device compute
            if N > MAX_THREAD:
                threadct =  MAX_THREAD, 1, 1
                blockct = int(ceil(float(N) / MAX_THREAD)), 1
            else:
                threadct =  N, 1, 1
                blockct  =  1, 1

            kernelargs = list(map(cudriver.In, bcargs))
            kernelargs += [cudriver.Out(retary), np.int32(N)]

            time = kernel(*kernelargs,
                          block=threadct, grid=blockct,
                          time_kernel=True)
            # print 'kernel time = %s' % time

            return retary

        return _ufunc_hack

    def _build_caller(self, lfunc):
        lfunc.calling_convention = CC_PTX_DEVICE
        lfunc.linkage = LINKAGE_INTERNAL       # do not emit device function
        lcaller_def = _CudaStagingCaller(CFuncRef(lfunc), lfunc.type.pointee)
        lcaller = lcaller_def(self.module)
        lcaller.calling_convention = CC_PTX_KERNEL
        return lcaller

