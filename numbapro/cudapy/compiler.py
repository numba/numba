import inspect
from numbapro.npm import symbolic, typing, codegen, execution
from numbapro.npm.compiler import get_func_name
from numbapro.npm.execution import (to_ctype, prepare_args, Complex64,
                                    Complex128, ArrayBase)
from numbapro.cudapipeline import nvvm, driver, devicearray
from .execution import CUDAKernel
from .typing import cudapy_typing_ext
from .codegen import cudapy_codegen_ext

CUDA_ADDR_SIZE = tuple.__itemsize__ * 8     # matches host

def compile_kernel(func, argtys):
    # symbolic interpretation
    se = symbolic.SymbolicExecution(func)
    se.visit()
    print se.dump()

    argspec = inspect.getargspec(func)
    assert not argspec.keywords
    assert not argspec.varargs
    assert not argspec.defaults

    # type infernece
    tydict = dict(zip(argspec.args, argtys))
    tydict[''] = None

    addrsize = CUDA_ADDR_SIZE
    globals = func.func_globals
    infer = typing.Infer(se.blocks, tydict, globals, intp=addrsize,
                         extended_globals=cudapy_typing_ext)

    typemap = infer.infer()

    # code generation
    name = get_func_name(func)
    cg = codegen.CodeGen(name, se.blocks, typemap, globals,
                         argtys, None, intp=addrsize,
                         extended_globals=cudapy_codegen_ext)
    lfunc = cg.generate()
    gvars = cg.extern_globals

    print lfunc

    lfunc.module.verify()

    # PTX-ization
    cudakernel = CUDAKernel(lfunc.name, to_ptx(lfunc), argtys)
    print cudakernel.ptx
    return cudakernel

def to_ptx(lfunc):
    context = driver.get_or_create_context()
    cc_major = context.device.COMPUTE_CAPABILITY[0]

    arch = 'compute_%d0' % cc_major

    nvvm.fix_data_layout(lfunc.module)
    nvvm.set_cuda_kernel(lfunc)
    ptx = nvvm.llvm_to_ptx(str(lfunc.module), opt=3, arch=arch)
    return ptx
