"""
Expose each GPU devices directly, OpenCL version
"""

from __future__ import print_function, absolute_import, division
from . import cl
from ... import servicelib
import functools
import weakref

platform = None
gpus = []

def init_gpus():
    """
    Populates global "gpus" as a list of GPU objects
    """
    global platform, gpus

    if gpus:
        assert len(gpus)
        return

    if platform is None:
        platform = cl.default_platform

    for idx, dev in enumerate(platform.all_devices):
        gpu = GPU(dev)
        gpus.append(gpu)
        globals()['gpu{0}'.format(idx)] = gpu


class GPU(object):
    """
    Proxy for an OpenCL compute device
    """
    def __init__(self, device):
        self._gpu = device
        self._context = None

    def __del__(self):
        del self._gpu
        del self._context

    def __getattr__(self, key):
        """
        redirect to self._gpu, filtering private attributes
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

    def push(self):
        self._context.push()

    def pop(self):
        self._context.pop()

    def __enter__(self):
        if get_context() is not self:
            self._context.push()
            _gpustack.push(self)

    def __exit__(self, exc_type, exc_val, exc_tb):
        assert get_context() is self
        self._context.pop()
        _gpustack.pop()

    def reset(self):
        if self._context:
            self._context.reset()
            self._context = None

def get_gpu(i):
    init_gpus()
    return gpus[i]

_gpu_stack = servicelib.TLStack()

def get_context(devnum=0):
    if not _gpustack:
        _gpustack.push(get_gpu(devnumb).context)

    return _gpustack.top

def require_context(fn):
    @functools.wraps(fn)
    def _require_cuda_context(*args, **kws):
        get_context()
        return fn(*args, **kws)

    return _require_cuda_context

def reset():
    for gpu in gpus:
        gpu.reset()

    _gpustack.clear()


# this cleanup function is needed to remove the list device objects. Otherwise
# the interpreter may fail cleanup of the list as the OpenCL driver could be
# deleted prior to the objects retained by this list.
#
# this seems a lighter weight to using weakproxies.
import atexit
def _cleanup():
    global gpus, platform
    for idx, _ in enumerate(platform.all_devices):
        del globals()['gpu{0}'.format(idx)]
    gpus = None
    platform = None

atexit.register(_cleanup)
