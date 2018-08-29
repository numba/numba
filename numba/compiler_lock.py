import threading
import functools


# Lock for the preventing multiple compiler execution
class _CompilerLock(object):
    def __init__(self):
        self._lock = threading.RLock()
        self._locked = 0

    def acquire(self):
        self._lock.acquire()
        self._locked += 1

    def release(self):
        self._locked -= 1
        self._lock.release()

    def __enter__(self):
        self.acquire()

    def __exit__(self, exc_val, exc_type, traceback):
        self.release()

    def is_locked(self):
        return self._locked > 0

    def __call__(self, func):
        @functools.wraps(func)
        def wrapped(*args, **kwargs):
            with self:
                return func(*args, **kwargs)
        return wrapped


global_compiler_lock = _CompilerLock()


def require_global_compiler_lock():
    """Sentry that checks the global_compiler_lock is acquired.
    """
    # Use assert to allow turning off this checks
    assert global_compiler_lock.is_locked()
