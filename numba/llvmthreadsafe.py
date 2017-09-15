"""
Utils for managing LLVM for threadsafe usage.
"""
import threading
import functools

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

