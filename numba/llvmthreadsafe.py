"""
Utils for managing LLVM for threadsafe usage.
"""
import types
import threading
import functools
import contextlib
import llvmlite.binding as llvm


class LockLLVM(object):
    """
    For locking LLVM.
    Usable as contextmanager and decorator.
    """

    def __init__(self):
        self._llvm_lock = threading.RLock()

    def __enter__(self):
        self._llvm_lock.acquire()

    def __exit__(self, *args, **kwargs):
        self._llvm_lock.release()

    def __call__(self, fn):
        @functools.wraps(fn)
        def wrapped(*args, **kwargs):
            with self:
                return fn(*args, **kwargs)
        return wrapped


# Make singleton
lock_llvm = LockLLVM()
del LockLLVM


def _patch_dispose(llvmobj):
    """
    Patch the _dispose method of the llvm object to use the llvm lock.
    """
    dtor = llvmobj._dispose

    def _ts_dispose():
        with lock_llvm:
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
parse_assembly = lock_llvm(_patch_retval_dispose(llvm.parse_assembly))
parse_bitcode = lock_llvm(_patch_retval_dispose(llvm.parse_bitcode))
create_mcjit_compiler = lock_llvm(_patch_retval_dispose(llvm.create_mcjit_compiler))
create_module_pass_manager = lock_llvm(_patch_retval_dispose(llvm.create_module_pass_manager))
create_function_pass_manager = lock_llvm(_patch_retval_dispose(llvm.create_function_pass_manager))
