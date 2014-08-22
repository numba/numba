"""
Expose each GPU devices directly
"""
from __future__ import print_function, absolute_import, division
import functools
from numba import servicelib
from .driver import driver


class _GPUListProxy(object):
    def __getitem__(self, item):
        init_gpus()
        return _gpus[item]


_gpus = []
gpus = _GPUListProxy()


def init_gpus():
    """
    Populates global "gpus" as a list of GPU objects
    """
    if _gpus:
        assert len(_gpus)
        return
    for num in range(driver.get_device_count()):
        device = driver.get_device(num)
        gpu = GPU(device)
        _gpus.append(gpu)
        globals()['gpu%d' % num] = gpu


class GPU(object):
    """Proxy into driver.Device
    """

    def __init__(self, gpu):
        self._gpu = gpu
        self._context = None

    def __getattr__(self, key):
        """Redirect to self._gpu
        """
        if key.startswith('_'):
            raise AttributeError(key)
        return getattr(self._gpu, key)

    def __repr__(self):
        return repr(self._gpu)

    @property
    def context(self):
        if self._context is None:
            self._context = self._gpu.create_context()
        return self._context

    def pop(self):
        self._context.pop()

    def push(self):
        self._context.push()

    def __enter__(self):
        if _get_device() is not self:
            # Don't push the context if it is the current device
            self.context.push()
            _gpustack.push(self)

    def __exit__(self, exc_type, exc_val, exc_tb):
        assert _get_device() is self
        if len(_gpustack) > 1:
            # Don't free the last the device on the stack
            self.context.pop()
            _gpustack.pop()

    def reset(self):
        if self._context:
            self._context.reset()
            self._context = None


def get_gpu(i):
    init_gpus()
    return _gpus[i]


_gpustack = servicelib.TLStack()


def _get_device(devnum=0):
    """Get the current device or use a device by device number.
    """
    if not _gpustack:
        _gpustack.push(get_gpu(devnum))
    return _gpustack.top


def get_context(devnum=0):
    """Get the current device or use a device by device number, and
    return the CUDA context.
    """
    return _get_device(devnum=devnum).context


def require_context(fn):
    """
    A decorator to ensure a context for the CUDA subsystem
    """

    @functools.wraps(fn)
    def _require_cuda_context(*args, **kws):
        get_context()
        return fn(*args, **kws)

    return _require_cuda_context


def reset():
    for gpu in _gpus:
        gpu.reset()
    _gpustack.clear()


