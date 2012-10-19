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
from numba.minivect import minitypes
from numbapro.translate import Translate as _Translate

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
    _threadIdx_x: 'llvm.nvvm.read.ptx.sreg.tid.x',
    _threadIdx_y: 'llvm.nvvm.read.ptx.sreg.tid.y',
    _threadIdx_z: 'llvm.nvvm.read.ptx.sreg.tid.z',

    _blockDim_x: 'llvm.nvvm.read.ptx.sreg.ntid.x',
    _blockDim_y: 'llvm.nvvm.read.ptx.sreg.ntid.y',
    _blockDim_z: 'llvm.nvvm.read.ptx.sreg.ntid.z',

    _blockIdx_x: 'llvm.nvvm.read.ptx.sreg.ctaid.x',
    _blockIdx_y: 'llvm.nvvm.read.ptx.sreg.ctaid.y',

    _gridDim_x: 'llvm.nvvm.read.ptx.sreg.nctaid.x',
    _gridDim_y: 'llvm.nvvm.read.ptx.sreg.nctaid.y',
}

_ATTRIBUTABLES = set([threadIdx, blockIdx, blockDim, gridDim, _THIS_MODULE])

def get_strided_arrays(argtypes):
    result = []
    for argtype in argtypes:
        if argtype.is_array:
            argtype = argtype.strided
        result.append(argtype)

    return result



# modify numba behavior
from numba import utils, functions, ast_translate
from numba import visitors, nodes, error, ast_type_inference
from numba import pipeline
import ast as _ast

class CudaAttributeNode(nodes.Node):
    def __init__(self, value):
        self.value = value

    def resolve(self, name):
        return type(self)(getattr(self.value, name))

    def __repr__(self):
        return '<%s value=%s>' % (type(self).__nmae__, self.value)

#class CudaDeviceCallRewrite(visitors.NumbaTransformer,
#                            ast_type_inference.NumpyMixin):
#    def visit_MathCallNode(self, ast):
#        name  = self.resolve_callee_name(ast.func)
#        func = self._myglobals[name]
#        return nodes.MathCallNode(func.signature, ast.args, func.lfunc, name=func.lfunc.name)

#    def resolve_callee_name(self, ast):
#        assert isinstance(ast, _ast.Name)
#        return ast.id

class CudaSRegRewrite(visitors.NumbaTransformer,
                      ast_type_inference.NumpyMixin):

    def visit_Attribute(self, ast):
        this_module = sys.modules[__name__]
        value = self.visit(ast.value)
        if isinstance(ast.value, _ast.Name):
            assert isinstance(value.ctx, _ast.Load)
            obj = self._myglobals.get(ast.value.id)
            if obj is this_module:
                retval = CudaAttributeNode(this_module).resolve(ast.attr)
        elif isinstance(value, CudaAttributeNode):
            retval = value.resolve(ast.attr)

        if retval.value in _SPECIAL_VALUES:
            # replace with a MathCallNode
            sig = minitypes.FunctionType(minitypes.uint32, [])
            fname = _SPECIAL_VALUES[retval.value]
            retval = nodes.MathCallNode(sig, [], None, name=fname)

        return retval


class NumbaproCudaPipeline(pipeline.Pipeline):
    def __init__(self, context, func, ast, func_signature, **kwargs):
        super(NumbaproCudaPipeline, self).__init__(context, func, ast,
                                               func_signature, **kwargs)
        self.insert_specializer('rewrite_cuda_sreg', after='type_infer')


    def rewrite_cuda_sreg(self, ast):
        return CudaSRegRewrite(self.context, self.func, ast).visit(ast)

context = utils.get_minivect_context()  # creates a new NumbaContext
context.llvm_context = ast_translate.LLVMContextManager()
context.numba_pipeline = NumbaproCudaPipeline
function_cache = context.function_cache = functions.FunctionCache(context)

cached = {}
def jit(restype=void, argtypes=None, backend='ast', **kws):
    '''JIT python function into a CUDA kernel.

    A CUDA kernel does not return any value.
    To retrieve result, use an array as a parameter.
    By default, all array data will be copied back to the host.
    Scalar parameter is copied to the host and will be not be read back.
    It is not possible to pass scalar parameter by reference.

    Support for double-precision floats depends on your CUDA device.
    '''

    assert argtypes is not None
    assert backend == 'ast', 'Bytecode support has dropped'

    #    restype = int32
    #    if backend=='bytecode':
    #        key = func, tuple(get_strided_arrays(argtypes))
    #        if key in cached:
    #            cnf = cached[key]
    #        else:
    #            # NOTE: This will use bytecode translate path
    #            t = CudaTranslate(func, restype=restype, argtypes=argtypes,
    #                              module=_lc.Module.new("ptx_%s" % str(func)))
    #            t.translate()

    #            cnf = CudaNumbaFunction(func, lfunc=t.lfunc, extra=t)
    #            cached[key] = cnf

    #        return cnf
    #    else:
    return jit2(restype=restype, argtypes=argtypes, **kws)


_device_functions = {}
def jit2(restype=void, argtypes=None, device=False, inline=False, **kws):
    assert device or restype == void,\
           ("Only device function can have return value %s" % restype)
    assert kws.get('_llvm_ee') is None
    assert not inline or device
    kws['nopython'] = True # override nopython option
    def _jit2(func):
        llvm_module = (kws.get('_llvm_module')
                       or _lc.Module.new('ptx_%s' % func))
        if not hasattr(func, '_is_numba_func'):
            func._is_numba_func = True
            func._numba_compile_only = True
        func._numba_inline = inline

        result = function_cache.compile_function(func, argtypes,
                                                 ctypes=True,
                                                 llvm_module=llvm_module,
                                                 llvm_ee=None,
                                                 **kws)

        signature, lfunc, unused = result

        # XXX: temp fix for PyIncRef and PyDecRef in lfunc.
        def _temp_hack():
            inlinelist = []
            fakepy = {}

            for bb in lfunc.basic_blocks:
                for instr in bb.instructions:
                    if isinstance(instr, _lc.CallOrInvokeInstruction):
                        fn = instr.called_function
                        fname = fn.name
                        if fname.startswith('Py_'):
                            inlinelist.append(instr)
                            fty = instr.called_function.type.pointee
                            fakepy[fname] = fn
                            assert fty.return_type == _lc.Type.void(),\
                                    'assume no sideeffect'

            for fname, fn in fakepy.items():
                bldr = _lc.Builder.new(fn.append_basic_block('entry'))
                bldr.ret_void()

            for call in inlinelist:
                assert _lc.inline_function(call)

            for fn in fakepy.values():
                fn.delete()

        _temp_hack()

        if device:
            assert lfunc.name not in _device_functions, 'Device function name already used'
            _device_functions[lfunc.name] = func, lfunc
            return CudaDeviceFunction(func, signature=signature, lfunc=lfunc)
        else:
            _link_device_function(lfunc)
            return CudaNumbaFunction(func, signature=signature, lfunc=lfunc)
    return _jit2


def _link_device_function(lfunc):
    toinline = []
    for bb in lfunc.basic_blocks:
        for instr in bb.instructions:
            if isinstance(instr, _lc.CallOrInvokeInstruction):
                fn = instr.called_function
                bag = _device_functions.get(fn.name)
                if bag is not None:
                    pyfunc, linkee =bag
                    lfunc.module.link_in(linkee.module.clone())
                    if pyfunc._numba_inline:
                        toinline.append(instr)

    for call in toinline:
        callee = call.called_function
        _lc.inline_function(call)


class CudaDeviceFunction(numba.decorators.NumbaFunction):
    def __init__(self, py_func, wrapper=None, ctypes_func=None,
                 signature=None, lfunc=None, extra=None):
        super(CudaDeviceFunction, self).__init__(py_func, wrapper, ctypes_func,
                                                signature, lfunc)

    def __call__(self, *args, **kws):
        raise TypeError("")

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

        func_name = lfunc.name

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

        print self.module

        from numbapro._cuda import nvvm
        nvvm.fix_data_layout(self.module)
        nvvm.set_cuda_kernel(lfunc)
        self._ptxasm = nvvm.llvm_to_ptx(str(self.module))
        print self._ptxasm
        from numbapro import _cudadispatch
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

    #    @property
    #    def compute_capability(self):
    #        '''The compute_capability of the PTX is generated for.
    #        '''
    #        return self._cc

    #    @property
    #    def target_machine(self):
    #        '''The LLVM target mcahine backend used to generate the PTX.
    #        '''
    #        return self._arch

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
                intr_func_bldr = _sreg(_SPECIAL_VALUES[res])
                intr = intr_func_bldr(self.mod)
                res = self.builder.call(intr, [])
            self.stack.append(_Variable(res))
        else:
            # fall back to default implementation
            super(CudaTranslate, self).op_LOAD_ATTR(i, op, arg)

# Patch numba
numba.decorators.jit_targets[('gpu', 'ast')] = jit2 # give up on bytecode path
numba.decorators.numba_function_autojit_targets['gpu'] = CudaAutoJitNumbaFunction
