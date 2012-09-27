import sys
from llvm import core as _lc
from llvm import ee as _le
from llvm import passes as _lp
from llvm_cbuilder import shortnames as _llvm_cbuilder_types
from numbapro.translate import Translate as _Translate
from numba.translate import Variable as _Variable
from numbapro import _cudadispatch

_THIS_MODULE = sys.modules[__name__]

# threadIdx

class _threadIdx_x: pass
class _threadIdx_y: pass
class _threadIdx_z: pass

class threadIdx:
    x = _threadIdx_x
    y = _threadIdx_y
    z = _threadIdx_z

# blockIdx

class _blockIdx_x: pass
class _blockIdx_y: pass

class blockIdx:
    x = _blockIdx_x
    y = _blockIdx_y

# blockDim

class _blockDim_x: pass
class _blockDim_y: pass
class _blockDim_z: pass

class blockDim:
    x = _blockDim_x
    y = _blockDim_y
    z = _blockDim_z

# gridDim

class _gridDim_x: pass
class _gridDim_y: pass

class gridDim:
    x = _gridDim_x
    y = _gridDim_y


_SPECIAL_VALUES = {
    _threadIdx_x: _lc.INTR_PTX_READ_TID_X,
    _threadIdx_y: _lc.INTR_PTX_READ_TID_Y,
    _threadIdx_z: _lc.INTR_PTX_READ_TID_Z,

    _blockDim_x: _lc.INTR_PTX_READ_NTID_X,
    _blockDim_y: _lc.INTR_PTX_READ_NTID_Y,
    _blockDim_z: _lc.INTR_PTX_READ_NTID_Z,

    _blockIdx_x: _lc.INTR_PTX_READ_CTAID_X,
    _blockIdx_y: _lc.INTR_PTX_READ_CTAID_Y,

    _gridDim_x: _lc.INTR_PTX_READ_NCTAID_X,
    _gridDim_y: _lc.INTR_PTX_READ_NCTAID_Y,
}

_ATTRIBUTABLES = set([threadIdx, blockIdx, blockDim, gridDim, _THIS_MODULE])

# decorators

__tr_map__ = {}

def jit(restype='int32', argtypes=[]):
    def _jit(func):
        global __tr_map__
        if func in __tr_map__:
            logger.warning("Warning: Previously compiled version of %r may be "
                           "garbage collected!" % (func,))

        #use_ast = False
        #if backend == 'ast':
        #    use_ast = True
        #    for arg_type in list(argtypes) + [restype]:
        #        if not isinstance(arg_type, minitypes.Type):
        #            use_ast = False
        #            debugout("String type specified, using bytecode translator...")
        #            break

        #if use_ast:
        #    return jit2(argtypes=argtypes)(func)
        #else:
        t = CudaTranslate(func, restype=restype, argtypes=argtypes)
        t.translate()

        cnf = CudaNumbaFunction(t.lfunc)

        __tr_map__[func] = cnf

        return cnf
    return _jit


class CudaNumbaFunction(object):
    _griddim = 1, 1, 1
    _blockdim = 1, 1, 1
    def __init__(self, lfunc):
        self.module = lfunc.module

        # builder wrapper
        wrapper_type = _lc.Type.function(_lc.Type.void(), lfunc.type.pointee.args)
        func_name = 'ptxwrapper_%s' % lfunc.name
        wrapper = self.module.add_function(wrapper_type, func_name)
        builder = _lc.Builder.new(wrapper.append_basic_block('entry'))
        inline_me = builder.call(lfunc, wrapper.args) # ignore return value
        builder.ret_void()

        # force inline of original function
        _lc.inline_function(inline_me)
        # then remove it
        lfunc.delete()

        wrapper.calling_convention = _lc.CC_PTX_KERNEL

        # optimize
        pmb = _lp.PassManagerBuilder.new()
        pmb.opt_level = 3

        pm = _lp.PassManager.new()

        pm.run(self.module)

        device_number = -1
        # TODO: Too be refacted. Copied from numbapro.vectorize.cuda
        cc = 'sm_%d%d' % _cudadispatch.compute_capability(device_number)
        self._cc = cc

        if _lc.HAS_PTX:
            arch = 'ptx%d' % _llvm_cbuilder_types.intp.width # select by host pointer size
        elif _lc.HAS_NVPTX:
            arch = {32: 'nvptx', 64: 'nvptx64'}[C.intp.width]
        else:
            raise Exception("llvmpy does not have PTX/NVPTX support")

        self._arch = arch

        assert _llvm_cbuilder_types.intp.width in [32, 64]

        ptxtm = _le.TargetMachine.lookup(arch, cpu=cc, opt=3)
        ptxasm = ptxtm.emit_assembly(self.module)
        self.ptxasm = ptxasm

        self.dispatcher = _cudadispatch.CudaNumbaFuncDispatcher(ptxasm,
                                                                func_name,
                                                                device_number)


    def configure(self, griddim, blockdim):
        self._griddim = griddim
        self._blockdim = blockdim

        while len(self._griddim) < 3:
            self._griddim += (1,)

        while len(self._blockdim) < 3:
            self._blockdim += (1,)


    def __call__(self, *args):
        self.dispatcher(args, self._griddim, self._blockdim)


    @property
    def compute_capability(self):
        return self._cc

    @property
    def architecture(self):
        return self._arch

class CudaTranslate(_Translate):
     def op_LOAD_ATTR(self, i, op, arg):
        '''
        Add cuda intrinsics
        '''
        peekarg = self.stack[-1]
        if peekarg.val in _ATTRIBUTABLES:
            objarg = self.stack.pop(-1)
            res = getattr(objarg.val, self.names[arg])
            if res in _SPECIAL_VALUES:
                intr_name = _SPECIAL_VALUES[res]
                intr = _lc.Function.intrinsic(self.mod, intr_name, [])
                res = self.builder.call(intr, [])
            self.stack.append(_Variable(res))
        else:
            # fall back to default implementation
            super(CudaTranslate, self).op_LOAD_ATTR(i, op, arg)

