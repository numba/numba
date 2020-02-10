from __future__ import print_function, absolute_import

import threading
import functools
import signal
from . import config
from .errors import FatalError

def _handle_SIGABRT(sig_no, frame):
    msg = ("The compilation chain has received SIGABRT. This is likely a "
           "problem emanating from LLVM itself. Rerun with the environment "
           "variable NUMBA_DEBUG_PERMIT_SIGABRT set to see the message from "
           "the system level signal handler.")
    raise FatalError(msg)


# Lock for the preventing multiple compiler execution, installs signal handlers
class _CompilerLock(object):
    def __init__(self):
        self._lock = threading.RLock()

    def acquire(self):
        self._lock.acquire()

    def release(self):
        self._lock.release()

    def __enter__(self):
        self.acquire()
        if not config.DEBUG_PERMIT_SIGABRT:
            self._curr_handler = signal.signal(signal.SIGABRT, _handle_SIGABRT)

    def __exit__(self, exc_val, exc_type, traceback):
        self.release()
        if not config.DEBUG_PERMIT_SIGABRT:
            signal.signal(signal.SIGABRT, self._curr_handler)

    def is_locked(self):
        is_owned = getattr(self._lock, '_is_owned')
        if not callable(is_owned):
            is_owned = self._is_owned
        return is_owned()

    def __call__(self, func):
        @functools.wraps(func)
        def _acquire_compile_lock(*args, **kwargs):
            with self:
                return func(*args, **kwargs)
        return _acquire_compile_lock

    def _is_owned(self):
        # This method is borrowed from threading.Condition.
        # Return True if lock is owned by current_thread.
        # This method is called only if _lock doesn't have _is_owned().
        if self._lock.acquire(0):
            self._lock.release()
            return False
        else:
            return True


global_compiler_lock = _CompilerLock()


def require_global_compiler_lock():
    """Sentry that checks the global_compiler_lock is acquired.
    """
    # Use assert to allow turning off this check
    assert global_compiler_lock.is_locked()
