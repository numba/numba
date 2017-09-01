import llvmlite.binding as llvm
import functools
from numba import llvmthreadsafe as llvmts

def _patch_dispose(llvmobj):
    """
    Patch the _dispose method of the llvm object to use the llvm lock.
    """
    dtor = llvmobj._dispose

    def _ts_dispose():
        dtor()

    llvmobj._dispose = _ts_dispose
    return llvmobj


def _patch_retval_dispose(fn):
    """
    Patch the _dispose method of the return value of the wrapped function.
    """
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        return _patch_dispose(fn(*args, **kwargs))
    return wrapper


# Bind llvm API with the lock
parse_assembly = llvmts.lock_llvm(_patch_retval_dispose(llvm.parse_assembly))
parse_bitcode = llvmts.lock_llvm(_patch_retval_dispose(llvm.parse_bitcode))
create_mcjit_compiler = llvmts.lock_llvm(llvm.create_mcjit_compiler)
create_module_pass_manager = llvmts.lock_llvm(llvm.create_module_pass_manager)
create_function_pass_manager = llvmts.lock_llvm(llvm.create_function_pass_manager)