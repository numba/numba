from llvm_cbuilder import *
from llvm_cbuilder import shortnames as C
from llvm.core import *
from llvm.passes import *
from llvm.ee import *
import numpy as np
from . import _common
from ._common import _llvm_ty_to_numpy

from numba.minivect import minitypes
from numbapro import _cudadispatch
from numbapro.translate import Translate
from numbapro.vectorize import minivectorize, basic

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

def get_dtypes(ret_type, arg_types):
    ret_dtype = minitypes.map_minitype_to_dtype(ret_type)
    arg_dtypes = tuple(minitypes.map_minitype_to_dtype(arg_type)
                           for arg_type in arg_types)
    return ret_dtype, arg_dtypes

class CudaVectorize(_common.GenericVectorize):
    def __init__(self, func):
        super(CudaVectorize, self).__init__(func)
        self.module = Module.new("ptx_%s" % func.func_name)
        self.signatures = []

    def add(self, ret_type, arg_types, **kwargs):
        kwargs.update({'module': self.module})
        self.signatures.append((ret_type, arg_types, kwargs))
        t = Translate(self.pyfunc, ret_type=ret_type, arg_types=arg_types,
                      **kwargs)
        t.translate()
        self.translates.append(t)

    def build_ufunc(self, device_number=-1):
        # quick & dirty tryout
        # PyCuda should be optional
        # from pycuda import driver as cudriver
        # from pycuda.autoinit import device, context # use default
        # from math import ceil

        lfunclist = self._get_lfunc_list()

        # setup optimizer for the staging caller
        fpm = FunctionPassManager.new(self.module)
        pmbldr = PassManagerBuilder.new()
        pmbldr.opt_level = 3
        pmbldr.populate(fpm)

        types_to_name = {}

        for (ret_type, arg_types, _), lfunc in zip(self.signatures, lfunclist):
            ret_dtype, arg_dtypes = get_dtypes(ret_type, arg_types)
            # generate a caller for all functions
            lcaller = self._build_caller(lfunc)
            fpm.run(lcaller)    # run the optimizer

            # unicode problem?
            fname = lcaller.name
            if isinstance(fname, unicode):
                fname = fname.encode('utf-8')

            types_to_name[arg_dtypes] = ret_dtype, fname

        # force inlining & trim internal functions
        pm = PassManager.new()
        pm.add(PASS_INLINE)
        pm.run(self.module)

        # generate ptx asm for all functions
        cc = 'compute_%d%d' % _cudadispatch.compute_capability(device_number)
        if HAS_PTX:
            arch = 'ptx%d' % C.intp.width # select by host pointer size
        elif HAS_NVPTX:
            arch = {32: 'nvptx', 64: 'nvptx64'}[C.intp.width]
        else:
            raise Exception("llvmpy does not have PTX/NVPTX support")

        assert C.intp.width in [32, 64]
        ptxtm = TargetMachine.lookup(arch, cpu=cc, opt=3) # TODO: ptx64 option
        ptxasm = ptxtm.emit_assembly(self.module)


        dispatcher = _cudadispatch.CudaUFuncDispatcher(ptxasm, types_to_name,
                                                       device_number)

        return minivectorize.fallback_vectorize(
                    basic.BasicVectorize, self.pyfunc, self.signatures,
                    minivect_dispatcher=None, cuda_dispatcher=dispatcher)


    def _build_caller(self, lfunc):
        lfunc.calling_convention = CC_PTX_DEVICE
        lfunc.linkage = LINKAGE_INTERNAL       # do not emit device function
        lcaller_def = _CudaStagingCaller(CFuncRef(lfunc), lfunc.type.pointee)
        lcaller = lcaller_def(self.module)
        lcaller.calling_convention = CC_PTX_KERNEL
        return lcaller

