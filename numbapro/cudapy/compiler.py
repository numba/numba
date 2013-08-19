import inspect
import llvm.core as lc
from numbapro.npm import (symbolic, typing, codegen, compiler, types, extending,
                          cgutils,)

from numbapro.cudadrv import nvvm, driver
from .execution import CUDAKernel
from . import ptxlib
#from .typing import cudapy_global_typing_ext, cudapy_call_typing_ext
#from .codegen import cudapy_global_codegen_ext, cudapy_call_codegen_ext
#from .passes import bind_scalar_constants

CUDA_ADDR_SIZE = tuple.__itemsize__ * 8     # matches host

def compile_kernel(func, argtys):
    lmod, lfunc, excs = compile_common(func, types.void, argtys,
                                       flags=['no-exceptions'])
    # PTX-ization
    wrapper = generate_kernel_wrapper(lfunc, bool(excs))
    cudakernel = CUDAKernel(wrapper.name, to_ptx(wrapper), argtys, excs)
    return cudakernel

def generate_kernel_wrapper(lfunc, has_excs):
    fname = '_cudapy_wrapper_' + lfunc.name
    argtypes = list(lfunc.type.pointee.args)
    if has_excs:
        exctype = lc.Type.int()
        argtypes.append(lc.Type.pointer(exctype)) # for exception
    fntype = lc.Type.function(lc.Type.void(), argtypes)

    wrapper = lfunc.module.add_function(fntype, fname)

    builder = lc.Builder.new(wrapper.append_basic_block(''))

    if has_excs:
        exc = builder.call(lfunc, wrapper.args[:-1])
    else:
        exc = builder.call(lfunc, wrapper.args)

    if has_excs:
        no_exc = lc.Constant.null(exctype)
        raised = builder.icmp(lc.ICMP_NE, exc, no_exc)
        
        with cgutils.if_then(builder, raised):
            builder.store(exc, wrapper.args[-1])
            builder.ret_void()

    builder.ret_void()

    lfunc.add_attribute(lc.ATTR_ALWAYS_INLINE)
    return wrapper

def compile_device(func, retty, argtys, inline=False):
    lmod, lfunc = compile_common(func, retty, argtys)
    if inline:
        lfunc.add_attribute(lc.ATTR_ALWAYS_INLINE)
    return DeviceFunction(func, lmod, lfunc, retty, argtys)

def declare_device_function(name, retty, argtys):
    lmod = lc.Module.new('extern-%s' % name)
    ts = codegen.TypeSetter(intp=CUDA_ADDR_SIZE)
    lret = ts.to_llvm(retty)
    largs = [ts.to_llvm(t) for t in argtys]
    lfty = lc.Type.function(lc.Type.void(), largs + [lc.Type.pointer(lret)])
    lfunc = lmod.add_function(lfty, name=name)
    edf = ExternalDeviceFunction(name, lmod, lfunc, retty, argtys)
    return edf

def get_cudapy_context():
    libs = compiler.get_builtin_context()
    extending.extends(libs, ptxlib.extensions)
    return libs

global_cudapy_libs = get_cudapy_context()

def compile_common(func, retty, argtys, flags=compiler.DEFAULT_FLAGS):
    libs = global_cudapy_libs
    lmod, lfunc, excs = compiler.compile_common(func, retty, argtys, libs=libs,
                                                flags=flags)
    return lmod, lfunc, excs

def to_ptx(lfunc):
    context = driver.get_or_create_context()
    cc_major, cc_minor = context.device.COMPUTE_CAPABILITY
    arch = nvvm.get_arch_option(cc_major, cc_minor)
    nvvm.fix_data_layout(lfunc.module)
    nvvm.set_cuda_kernel(lfunc)
    ptx = nvvm.llvm_to_ptx(str(lfunc.module), opt=3, arch=arch)
    return ptx

class DeviceFunction(object):
    def __init__(self, func, lmod, lfunc, retty, argtys):
        self.func = func
        self.args = tuple(argtys)
        self.return_type = retty
        self._npm_context_ = lmod, lfunc, self.return_type, self.args

    def __repr__(self):
        args = (self.return_type or 'void', self.args)
        return '<cuda device function %s%s>' % args

class ExternalDeviceFunction(object):
    def __init__(self, name, lmod, lfunc, retty, argtys):
        self.name = name
        self.args = tuple(argtys)
        self.return_type = retty
        self._npm_context_ = lmod, lfunc, self.return_type, self.args

    def __repr__(self):
        args = (self.name, self.return_type or 'void', self.args)
        return '<cuda external device function %s %s%s>' % args
