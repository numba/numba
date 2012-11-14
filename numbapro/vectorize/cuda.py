from llvm_cbuilder import *
from llvm_cbuilder import shortnames as C
from llvm.core import *
from llvm.passes import *
from llvm.ee import *
import numpy as np
from . import _common
from ._common import _llvm_ty_to_numpy

from numba.minivect import minitypes

from numbapro._cuda.error import CudaSupportError
from numbapro._cuda import nvvm
try:
    from numbapro import _cudadispatch
except CudaSupportError: # ignore missing cuda dependency
    pass

from numbapro.translate import Translate
from numbapro.vectorize import minivectorize, basic


class _CudaStagingCaller(CDefinition):
    def body(self, *args, **kwargs):
        worker = self.depends(self.WorkerDef)
        inputs = args[:-2]
        output, ct = args[-2:]

        # get current thread index
        fty_sreg = Type.function(Type.int(), [])
        def get_ptx_sreg(name):
            m = self.function.module
            prefix = 'llvm.nvvm.read.ptx.sreg.'
            return CFunc(self, m.get_or_insert_function(fty_sreg,
                                                        name=prefix + name))

        tid_x = get_ptx_sreg('tid.x')
        ntid_x = get_ptx_sreg('ntid.x')
        ctaid_x = get_ptx_sreg('ctaid.x')

        tid = self.var_copy(tid_x())
        blkdim = self.var_copy(ntid_x())
        blkid = self.var_copy(ctaid_x())

        i = tid + blkdim * blkid
        with self.ifelse( i >= ct ) as ifelse: # stop condition
            with ifelse.then():
                self.ret()

        res = worker(*map(lambda x: x[i], inputs), inline=True)
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

def get_dtypes(restype, argtypes):
    try:
        ret_dtype = minitypes.map_minitype_to_dtype(restype)
    except KeyError:
        ret_dtype = None

    arg_dtypes = tuple(minitypes.map_minitype_to_dtype(arg_type)
                           for arg_type in argtypes)
    return ret_dtype, arg_dtypes

class CudaVectorize(_common.GenericVectorize):
    def __init__(self, func):
        super(CudaVectorize, self).__init__(func)
        self.module = Module.new("ptx_%s" % func.func_name)
        self.signatures = []

    def add(self, restype, argtypes, **kwargs):
        kwargs.update({'module': self.module})
        self.signatures.append((restype, argtypes, kwargs))
        t = Translate(self.pyfunc, restype=restype, argtypes=argtypes,
                      **kwargs)
        t.translate()
        self.translates.append(t)
        self.args_restypes.append(argtypes + [restype])

    def _build_ufunc(self):
        lfunclist = self._get_lfunc_list()

        # setup optimizer for the staging caller
        fpm = FunctionPassManager.new(self.module)
        pmbldr = PassManagerBuilder.new()
        pmbldr.opt_level = 3
        pmbldr.populate(fpm)

        types_to_name = {}

        lcaller_list = []
        for (restype, argtypes, _), lfunc in zip(self.signatures, lfunclist):
            ret_dtype, arg_dtypes = get_dtypes(restype, argtypes)
            # generate a caller for all functions
            lcaller = self._build_caller(lfunc)
            assert lcaller.module is lfunc.module
            fpm.run(lcaller)    # run the optimizer
            lcaller_list.append(lcaller)

            # unicode problem?
            fname = lcaller.name
            if isinstance(fname, unicode):
                fname = fname.encode('utf-8')

            types_to_name[arg_dtypes] = ret_dtype, fname

        #        # force inlining & trim internal functions
        #        pm = PassManager.new()
        #        pm.add(PASS_INLINE)
        #        pm.run(self.module)

        #        # generate ptx asm for all functions

        #        # Note. Oddly, the llvm ptx backend does not have compute capacility
        #        #       beyound 2.0, but it has the streaming-multiprocessor,
        #        #       which is the same.
        #        cc = 'sm_%d%d' % _cudadispatch.compute_capability()
        #        if HAS_PTX:
        #            arch = 'ptx%d' % C.intp.width # select by host pointer size
        #        elif HAS_NVPTX:
        #            arch = {32: 'nvptx', 64: 'nvptx64'}[C.intp.width]
        #        else:
        #            raise Exception("llvmpy does not have PTX/NVPTX support")

        #        assert C.intp.width in [32, 64]
        #        ptxtm = TargetMachine.lookup(arch, cpu=cc, opt=3) # TODO: ptx64 option
        #        ptxasm = ptxtm.emit_assembly(self.module)


        nvvm.fix_data_layout(self.module)
        for lfunc in lcaller_list:
            nvvm.set_cuda_kernel(lfunc)

        for lfunc in lfunclist:
            lfunc.delete()

        ptxasm = nvvm.llvm_to_ptx(str(self.module))
        dispatcher = _cudadispatch.CudaUFuncDispatcher(ptxasm, types_to_name)
        return dispatcher

    def build_ufunc(self):
        dispatcher = self._build_ufunc()
        ufunc, lfuncs = minivectorize.fallback_vectorize(
                    basic.BasicVectorize, self.pyfunc, self.signatures,
                    minivect_dispatcher=None, cuda_dispatcher=dispatcher)
        return ufunc

    def build_ufunc_core(self):
        # TODO: implement this after the refactoring of cuda dispatcher
        raise NotImplementedError

    def _build_caller(self, lfunc):
        lcaller_def = _CudaStagingCaller(CFuncRef(lfunc), lfunc.type.pointee)
        lcaller = lcaller_def(self.module)
        return lcaller
