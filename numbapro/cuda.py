import sys
import logging
logger = logging.getLogger(__name__)

from llvm import core as _lc
from llvm import ee as _le
from llvm import passes as _lp
from llvm_cbuilder import shortnames as _llvm_cbuilder_types

from numba import *
import numba.decorators
from numba.translate import Variable as _Variable

from numbapro.translate import Translate as _Translate
from numbapro import _cudadispatch
from numbapro._cuda.default import device as _cuda_device
from numbapro._cuda import nvvm


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

### The Old way with LLVM PTX and NVPTX
#_SPECIAL_VALUES = {
#    _threadIdx_x: _lc.INTR_PTX_READ_TID_X,
#    _threadIdx_y: _lc.INTR_PTX_READ_TID_Y,
#    _threadIdx_z: _lc.INTR_PTX_READ_TID_Z,

#    _blockDim_x: _lc.INTR_PTX_READ_NTID_X,
#    _blockDim_y: _lc.INTR_PTX_READ_NTID_Y,
#    _blockDim_z: _lc.INTR_PTX_READ_NTID_Z,

#    _blockIdx_x: _lc.INTR_PTX_READ_CTAID_X,
#    _blockIdx_y: _lc.INTR_PTX_READ_CTAID_Y,

#    _gridDim_x: _lc.INTR_PTX_READ_NCTAID_X,
#    _gridDim_y: _lc.INTR_PTX_READ_NCTAID_Y,
#}

def _sreg(name):
    def wrap(module):
        fty_sreg =_lc.Type.function(_lc.Type.int(), [])
        return module.get_or_insert_function(fty_sreg, name=name)
    return wrap

_SPECIAL_VALUES = {
    _threadIdx_x: _sreg('llvm.nvvm.read.ptx.sreg.tid.x'),
    _threadIdx_y: _sreg('llvm.nvvm.read.ptx.sreg.tid.y'),
    _threadIdx_z: _sreg('llvm.nvvm.read.ptx.sreg.tid.z'),

    _blockDim_x: _sreg('llvm.nvvm.read.ptx.sreg.ntid.x'),
    _blockDim_y: _sreg('llvm.nvvm.read.ptx.sreg.ntid.y'),
    _blockDim_z: _sreg('llvm.nvvm.read.ptx.sreg.ntid.z'),

    _blockIdx_x: _sreg('llvm.nvvm.read.ptx.sreg.ctaid.x'),
    _blockIdx_y: _sreg('llvm.nvvm.read.ptx.sreg.ctaid.y'),

    _gridDim_x: _sreg('llvm.nvvm.read.ptx.sreg.nctaid.x'),
    _gridDim_y: _sreg('llvm.nvvm.read.ptx.sreg.nctaid.y'),
}

_ATTRIBUTABLES = set([threadIdx, blockIdx, blockDim, gridDim, _THIS_MODULE])

def get_strided_arrays(argtypes):
    result = []
    for argtype in argtypes:
        if argtype.is_array:
            argtype = argtype.strided
        result.append(argtype)

    return result

# decorators
cached = {}
def jit(restype=void, argtypes=None, backend='bytecode'):
    '''JIT python function into a CUDA kernel.

    A CUDA kernel does not return any value.
    To retrieve result, use an array as a parameter.
    By default, all array data will be copied back to the host.
    Scalar parameter is copied to the host and will be not be read back.
    It is not possible to pass scalar parameter by reference.

    Support for double-precision floats depends on your CUDA device.
    '''
    assert restype == void, restype
    assert argtypes is not None
    assert backend == 'bytecode'

    restype = int32

    def _jit(func):
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

        key = func, tuple(get_strided_arrays(argtypes))
        if key in cached:
            cnf = cached[key]
        else:
            # NOTE: This will use bytecode translate path
            t = CudaTranslate(func, restype=restype, argtypes=argtypes,
                              module=_lc.Module.new("ptx_%s" % str(func)))
            t.translate()

            cnf = CudaNumbaFunction(func, lfunc=t.lfunc, extra=t)
            cached[key] = cnf

        return cnf

    return _jit


class CudaBaseFunction(numba.decorators.NumbaFunction):

    _griddim = 1, 1, 1      # default grid dimension
    _blockdim = 1, 1, 1     # default block dimension

    def configure(self, griddim, blockdim):
        '''Returns a new instance that is configured with the
        specified kernel grid dimension and block dimension.

        griddim, blockdim -- Triples of at most 3 integers. Missing dimensions
                             are automatically filled with `1`.

        '''
        import copy
        inst = copy.copy(self) # clone the object

        inst._griddim = griddim
        inst._blockdim = blockdim

        while len(inst._griddim) < 3:
            inst._griddim += (1,)

        while len(inst._blockdim) < 3:
            inst._blockdim += (1,)

        return inst

    def __getitem__(self, args):
        '''Shorthand for self.configure()
        '''
        griddim, blockdim = args
        return self.configure(griddim, blockdim)


class CudaNumbaFunction(CudaBaseFunction):

    def __init__(self, py_func, wrapper=None, ctypes_func=None,
                 signature=None, lfunc=None, extra=None):
        super(CudaNumbaFunction, self).__init__(py_func, wrapper, ctypes_func,
                                                signature, lfunc)
        # print 'translating...'
        self.module = lfunc.module
        self.extra = extra

        def apply_typemap(ty):
            if isinstance(ty, _lc.IntegerType):
                return 'i'
            elif ty == _lc.Type.float():
                return 'f'
            elif ty == _lc.Type.double():
                return 'd'
            else:
                return '_'

        argtys = lfunc.type.pointee.args
        typemap_string = ''.join(map(apply_typemap, argtys))

        self.typemap = typemap_string

        # builder wrapper
        wrapper_type = _lc.Type.function(_lc.Type.void(), argtys)
        func_name = 'ptxwrapper_%s' % lfunc.name
        wrapper = self.module.add_function(wrapper_type, func_name)
        builder = _lc.Builder.new(wrapper.append_basic_block('entry'))

        inline_me = builder.call(lfunc, wrapper.args) # ignore return value
        builder.ret_void()

        # force inline of original function
        _lc.inline_function(inline_me)
        # then remove it
        lfunc.delete()

        #### The Old Way that uses LLVM PTX and NVPTX
        #    wrapper.calling_convention = _lc.CC_PTX_KERNEL

        #    # optimize
        #    pmb = _lp.PassManagerBuilder.new()
        #    pmb.opt_level = 3

        #    pm = _lp.PassManager.new()

        #    pm.run(self.module)

        #    device_number = -1
        #    # TODO: Too be refacted. Copied from numbapro.vectorize.cuda
        #    cc = 'sm_%d%d' % _cuda_device.COMPUTE_CAPABILITY
        #    self._cc = cc

        #    if _lc.HAS_PTX:
        #        arch = 'ptx%d' % _llvm_cbuilder_types.intp.width # select by host pointer size
        #    elif _lc.HAS_NVPTX:
        #        arch = {32: 'nvptx', 64: 'nvptx64'}[C.intp.width]
        #    else:
        #        raise Exception("llvmpy does not have PTX/NVPTX support")

        #    self._arch = arch

        #    assert _llvm_cbuilder_types.intp.width in [32, 64]

        #    # generate PTX
        #    ptxtm = _le.TargetMachine.lookup(arch, cpu=cc, opt=3)
        #    ptxasm = ptxtm.emit_assembly(self.module)
        #    self._ptxasm = ptxasm

        nvvm.fix_data_layout(self.module)
        nvvm.set_cuda_kernel(wrapper)
        self._ptxasm = nvvm.llvm_to_ptx(str(self.module))
        self.dispatcher = _cudadispatch.CudaNumbaFuncDispatcher(self._ptxasm,
                                                                func_name,
                                                                typemap_string)

    def __call__(self, *args):
        '''Call the CUDA kernel.

        This call is synchronous to the host.
        In another words, this function will return only upon the completion
        of the CUDA kernel.
        '''
        # Cast scalar arguments to match the prototype.
        def convert(ty, val):
            if ty == 'f' or ty == 'd':
                return float(val)
            elif ty == 'i':
                return int(val)
            else:
                return val

        args = [convert(ty, val) for ty, val in zip(self.typemap, args)]
        self.dispatcher(args, self._griddim, self._blockdim)

    @property
    def compute_capability(self):
        '''The compute_capability of the PTX is generated for.
        '''
        return self._cc

    @property
    def target_machine(self):
        '''The LLVM target mcahine backend used to generate the PTX.
        '''
        return self._arch

    @property
    def ptx(self):
        '''Returns the PTX assembly for this function.
        '''
        return self._ptxasm

class CudaAutoJitNumbaFunction(CudaBaseFunction):

    def invoke_compiled(self, compiled_cuda_func, *args, **kwargs):
        return compiled_cuda_func[self._griddim, self._blockdim](*args, **kwargs)

class CudaTranslate(_Translate):
     def op_LOAD_ATTR(self, i, op, arg):
        '''Add cuda intrinsics lookup.
        '''
        peekarg = self.stack[-1]
        if peekarg.val in _ATTRIBUTABLES:
            objarg = self.stack.pop(-1)
            res = getattr(objarg.val, self.names[arg])
            if res in _SPECIAL_VALUES:
                intr_func_bldr = _SPECIAL_VALUES[res]
                intr = intr_func_bldr(self.mod)
                res = self.builder.call(intr, [])
            self.stack.append(_Variable(res))
        else:
            # fall back to default implementation
            super(CudaTranslate, self).op_LOAD_ATTR(i, op, arg)

# Patch numba
numba.decorators.jit_targets['gpu'] = jit
numba.decorators.numba_function_autojit_targets['gpu'] = CudaAutoJitNumbaFunction
