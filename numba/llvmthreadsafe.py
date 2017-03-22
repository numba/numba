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


def _set_dispose(dtor):
    def _ts_dispose():
        with lock_llvm:
            dtor()
    return _ts_dispose


# Bind llvm API with the lock
@lock_llvm
def parse_assembly(*args, **kwargs):
    mod = llvm.parse_assembly(*args, **kwargs)
    mod._dispose = _set_dispose(mod._dispose)
    return mod


@lock_llvm
def parse_bitcode(*args, **kwargs):
    mod = llvm.parse_bitcode(*args, **kwargs)
    mod._dispose = _set_dispose(mod._dispose)
    return mod


create_mcjit_compiler = lock_llvm(llvm.create_mcjit_compiler)
create_module_pass_manager = lock_llvm(llvm.create_module_pass_manager)
create_function_pass_manager = lock_llvm(llvm.create_function_pass_manager)
