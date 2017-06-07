"""
Expose each GPU devices directly, OpenCL version
"""

from __future__ import print_function, absolute_import, division
import functools
import weakref
from ... import servicelib
from .driver import driver

platform = None
_ocl_context = None
_gpustack = None
gpus = []

def _init_gpus():
    """
    Populates global "gpus" as a list of GPU objects
    This can be seen as a lazy constructor for the module
    """
    global platform, gpus, _ocl_context, _gpustack

    if gpus:
        assert len(gpus)
        return

    if platform is None:
        platform = driver.default_platform

    _gpustack = servicelib.TLStack()
    all_devs = platform.all_devices
    _ocl_context = driver.create_context(platform, all_devs)

    for idx, dev in enumerate(all_devs):
        gpu = GPU(dev)
        gpus.append(gpu)
        globals()['gpu{0}'.format(idx)] = gpu

def _cleanup_gpus():
    """
    Destructor for the module
    """
    global gpus, platform, _ocl_context, _gpustack
    for idx in range(len(gpus)):
        del globals()['gpu{0}'.format(idx)]
    gpus = None
    _ocl_context = None
    _gpustack = None
    platform = None


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
        """
        In OpenCL this context is actually a queue associated to this
        device
        """
        if self._context is None:
            try:
                self._context = _ocl_context.create_command_queue(self._gpu)
            except AttributeError as e:
                print (e)

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
        self._context = None

def get_ocl_context():
    _init_gpus()
    return _ocl_context

def get_gpu(i):
    _init_gpus()
    return gpus[i]



def get_context(devnum=0):
    _init_gpus()
    if not _gpustack:
        _gpustack.push(get_gpu(devnum).context)

    return _gpustack.top

def require_context(fn):
    @functools.wraps(fn)
    def _require_ocl_context(*args, **kws):
        get_context()
        return fn(*args, **kws)

    return _require_ocl_context

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

atexit.register(_cleanup_gpus) #register destructor