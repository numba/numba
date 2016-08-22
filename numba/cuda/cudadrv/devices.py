"""
Expose each GPU devices directly.

This module implements a API that is like the "CUDA runtime" context manager
for managing CUDA context stack and clean up.  It relies on thread-local globals
to separate the context stack management of each thread. Contexts are also
sharable among threads.  Only the main thread can destroy Contexts.

Note:
- This module must be imported by the main-thread.

"""
from __future__ import print_function, absolute_import, division
import functools
import threading
from numba import servicelib
from .driver import driver


class _DeviceList(object):
    def __getattr__(self, attr):
        # First time looking at "lst" attribute.
        if attr == "lst":
            # Device list is not initialized.
            # Query all CUDA devices.
            numdev = driver.get_device_count()
            gpus = [_DeviceContextManager(driver.get_device(devid))
                    for devid in range(numdev)]
            # Define "lst" to avoid re-initialization
            self.lst = gpus
            return gpus

        # Other attributes
        return super(_DeviceList, self).__getattr__(attr)

    def __getitem__(self, devnum):
        '''
        Returns the context manager for device *devnum*.
        '''
        return self.lst[devnum]

    def __str__(self):
        return ', '.join([str(d) for d in self.lst])

    def __iter__(self):
        return iter(self.lst)

    def __len__(self):
        return len(self.lst)

    @property
    def current(self):
        """Returns the active device or None if there's no active device
        """
        if _runtime.context_stack:
            return self.lst[_runtime.current_context.device.id]


class _DeviceContextManager(object):
    """
    Provides a context manager for executing in the context of the chosen
    device. The normal use of instances of this type is from
    ``numba.cuda.gpus``. For example, to execute on device 2::

       with numba.cuda.gpus[2]:
           d_a = numba.cuda.to_device(a)

    to copy the array *a* onto device 2, referred to by *d_a*.
    """

    def __init__(self, device):
        self._device = device

    def __getattr__(self, item):
        return getattr(self._device, item)

    def __enter__(self):
        _runtime.push_context(self)

    def __exit__(self, exc_type, exc_val, exc_tb):
        _runtime.pop_context()

    def __str__(self):
        return "<Managed Device {self.id}>".format(self=self)


class _Runtime(object):
    """Emulate the CUDA runtime context management.

    It owns all Devices and Contexts.
    Keeps at most one Context per Device
    """

    def __init__(self):
        self.gpus = _DeviceList()

        # A thread local stack
        self.context_stack = servicelib.TLStack()

        # Remember the main thread
        # Only the main thread can *actually* destroy
        self._mainthread = threading.current_thread()

        # Avoid mutation of runtime state in multithreaded programs
        self._lock = threading.RLock()

    @property
    def current_context(self):
        """Return the active gpu context
        """
        return self.context_stack.top

    def _get_or_create_context(self, gpu):
        """Try to use a already created context for the given gpu.  If none
        existed, create a new context.

        Returns the context
        """
        with self._lock:
            ctx = gpu.get_primary_context()
            ctx.push()
            return ctx

    def push_context(self, gpu):
        """Push a context for the given GPU or create a new one if no context
        exist for the given GPU.
        """
        # Context stack is empty or the active device is not the given gpu
        if self.context_stack.is_empty or self.current_context.device != gpu:
            ctx = self._get_or_create_context(gpu)

        # Active context is from the gpu
        else:
            ctx = self.current_context

        # Always put the new context on the stack
        self.context_stack.push(ctx)
        return ctx

    def pop_context(self):
        """Pop a context from the context stack if there is more than
        one context in the stack.

        Will not remove the last context in the stack.
        """
        ctx = self.current_context
        # If there is more than one context
        # Do not pop the last context so there is always a active context
        if len(self.context_stack) > 1:
            ctx.pop()
            self.context_stack.pop()
        assert self.context_stack

    def get_or_create_context(self, devnum):
        """Returns the current context or push/create a context for the GPU
        with the given device number.
        """
        if self.context_stack:
            return self.current_context
        else:
            with self._lock:
                return self.push_context(self.gpus[devnum])

    def reset(self):
        """Clear all contexts in the thread.  Destroy the context if and only
        if we are in the main thread.
        """
        # Clear the context stack
        while self.context_stack:
            ctx = self.context_stack.pop()
            ctx.pop()

        # If it is the main thread
        if threading.current_thread() == self._mainthread:
            self._destroy_all_contexts()

    def _destroy_all_contexts(self):
        # Reset all devices
        for gpu in self.gpus:
            gpu.reset()


_runtime = _Runtime()

# ================================ PUBLIC API ================================

gpus = _runtime.gpus


def get_context(devnum=0):
    """Get the current device or use a device by device number, and
    return the CUDA context.
    """
    return _runtime.get_or_create_context(devnum)


def require_context(fn):
    """
    A decorator that ensures a CUDA context is available when *fn* is executed.

    Decorating *fn* is equivalent to writing::

       get_context()
       fn()

    at each call site.
    """

    @functools.wraps(fn)
    def _require_cuda_context(*args, **kws):
        get_context()
        return fn(*args, **kws)

    return _require_cuda_context


def reset():
    """Reset the CUDA subsystem for the current thread.

    In the main thread:
    This removes all CUDA contexts.  Only use this at shutdown or for
    cleaning up between tests.

    In non-main threads:
    This clear the CUDA context stack only.

    """
    _runtime.reset()
