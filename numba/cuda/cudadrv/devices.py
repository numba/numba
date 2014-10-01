"""
Expose each GPU devices directly
"""
from __future__ import print_function, absolute_import, division
import functools
from numba import servicelib
from .driver import driver
import threading


class _gpus(object):
    """A thread local list of GPU instances
    """

    def __init__(self):
        self._tls = threading.local()

    @property
    def _gpus(self):
        try:
            return self._tls.gpus
        except AttributeError:
            self._tls.gpus = self._init_gpus()
            return self._tls.gpus

    def _init_gpus(self):
        gpus = []
        for num in range(driver.get_device_count()):
            device = driver.get_device(num)
            gpus.append(GPU(device))
        return gpus

    def __getitem__(self, item):
        return self._gpus[item]

    def append(self, item):
        return self._gpus.append(item)

    def __len__(self):
        return len(self._gpus)

    def __nonzero__(self):
        return bool(self._gpus)

    def __iter__(self):
        return iter(self._gpus)

    __bool__ = __nonzero__

    def reset(self):
        for gpu in self:
            gpu.reset()


gpus = _gpus()
del _gpus


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
        if self._context is None:
            self.context
        else:
            self._context.push()
        _gpustack.push(self)

    def __exit__(self, exc_type, exc_val, exc_tb):
        assert _get_device() is self
        self.context.pop()
        _gpustack.pop()

    def reset(self):
        if self._context:
            self._context.reset()
            self._context = None


def get_gpu(i):
    return gpus[i]


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
    gpus.reset()
    _gpustack.clear()


