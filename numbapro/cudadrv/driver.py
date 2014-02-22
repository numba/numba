"""
CUDA driver bridge implementation
"""

from __future__ import absolute_import, print_function, division
import sys
import os
import traceback
import ctypes
import weakref
import functools
from ctypes import (c_int, byref, c_size_t, c_char, c_char_p, addressof,
                    c_void_p)
import contextlib
from numba import utils
from numbapro import servicelib
from numbapro.cudadrv.error import CudaSupportError, CudaDriverError
from .drvapi import API_PROTOTYPES
from . import enums, drvapi


VERBOSE_JIT_LOG = int(os.environ.get('NUMBAPRO_VERBOSE_CU_JIT_LOG', 1))
MIN_REQUIRED_CC = (2, 0)


def find_driver():
    envpath = os.environ.get('NUMBAPRO_CUDA_DRIVER', None)
    if envpath is not None:
        try:
            envpath = os.path.abspath(envpath)
        except ValueError:
            raise ValueError("NUMBAPRO_CUDA_DRIVER %s is not a valid path" %
                             envpath)
        if not os.path.isfile(envpath):
            raise ValueError("NUMBAPRO_CUDA_DRIVER %s is not a valid file "
                             "path.  Note it must be a filepath of the .so/"
                             ".dll/.dylib or the driver" % envpath)
        candidates = [envpath]
    else:
        # Determine DLL type
        if sys.platform == 'win32':
            dlloader = ctypes.WinDLL
            dldir = ['\\windows\\system32']
            dlname = 'nvcuda.dll'
        elif sys.platform == 'darwin':
            dlloader = ctypes.CDLL
            dldir = ['/usr/local/cuda/lib']
            dlname = 'libcuda.dylib'
        else:
            # Assume to be *nix like
            dlloader = ctypes.CDLL
            dldir = ['/usr/lib', '/usr/lib64']
            dlname = 'libcuda.so'

        # First search for the name in the default library path.
        # If that is not found, try the specific path.
        candidates = [dlname] + [os.path.join(x, dlname) for x in dldir]

    # Load the driver; Collect driver error information
    path_not_exist = []
    driver_load_error = []

    for path in candidates:
        try:
            dll = dlloader(path)
        except OSError as e:
            # Problem opening the DLL
            path_not_exist.append(not os.path.isfile(path))
            driver_load_error.append(e)
        else:
            return dll

    # Problem loading driver
    if all(path_not_exist):
        _raise_driver_not_found()
    else:
        errmsg = '\n'.join(str(e) for e in driver_load_error)
        _raise_driver_error(errmsg)


DRIVER_NOT_FOUND_MSG = """
CUDA driver library cannot be found.
If you are sure that a CUDA driver is installed,
try setting environment variable NUMBAPRO_CUDA_DRIVER
with the file path of the CUDA driver shared library.
"""


DRIVER_LOAD_ERROR_MSG = """
Possible CUDA driver libraries are found but error occurred during load:
%s
"""


def _raise_driver_not_found():
    raise CudaSupportError(DRIVER_NOT_FOUND_MSG)


def _raise_driver_error(e):
    raise CudaSupportError(DRIVER_LOAD_ERROR_MSG % e)


def _build_reverse_error_map():
    prefix = 'CUDA_ERROR'
    map = utils.UniqueDict()
    for name in dir(enums):
        if name.startswith(prefix):
            code = getattr(enums, name)
            map[code] = name
    return map


ERROR_MAP = _build_reverse_error_map()


class Driver(object):
    """
    Driver API functions are lazily bound.
    """
    _singleton = None

    def __new__(cls):
        obj = cls._singleton
        if obj is not None:
            return obj
        else:
            obj = object.__new__(cls)
            obj.lib = find_driver()
            obj.cuInit(0)
            cls._singleton = obj
        return obj

    def __init__(self):
        self.devices = utils.UniqueDict()

    def __getattr__(self, fname):
        # First request of a driver API function
        try:
            proto = API_PROTOTYPES[fname]
        except KeyError:
            raise AttributeError(fname)
        restype = proto[0]
        argtypes = proto[1:]
        try:
            # Try newer API
            libfn = getattr(self.lib, fname + "_v2")
        except AttributeError:
            libfn = getattr(self.lib, fname)
        libfn.restype = restype
        libfn.argtypes = argtypes

        @functools.wraps(libfn)
        def wrapper(*args):
            retcode = libfn(*args)
            self._check_error(fname, retcode)

        setattr(self, fname, wrapper)
        return wrapper

    def _check_error(self, fname, retcode):
        if retcode != enums.CUDA_SUCCESS:
            errname = ERROR_MAP.get(retcode, "UNKNOWN_CUDA_ERROR")
            raise CudaDriverError("Call to %s results in %s" % (fname,
                                                                errname))

    def get_device(self, devnum=0):
        dev = self.devices.get(devnum)
        if dev is None:
            dev = Device(devnum)
            self.devices[devnum] = dev
        return weakref.proxy(dev)

    def get_num_devices(self):
        count = c_int()
        self.cuDeviceGetCount(byref(count))
        return count.value

    def list_devices(self):
        """Returns a list of active devices
        """
        return list(self.devices.values())

    def reset(self):
        """Reset all devices
        """
        for dev in self.devices.values():
            dev.reset()


driver = Driver()


class TrashService(servicelib.Service):
    """
    We need this to enqueue things to be removed.  There are times when you
    want to disable deallocation because that would break asynchronous work
    queues.
    """
    CLEAN_LIMIT = 20

    def add_trash(self, item):
        self.trash.append(item)

    def process(self, _arg):
        self.trash = []
        yield
        while True:
            count = 0
            # Clean the trash
            assert self.CLEAN_LIMIT > count
            while self.trash and count < self.CLEAN_LIMIT:
                cb = self.trash.pop()
                # Invoke callback
                cb()
                count += 1
            yield

    def clear(self):
        while self.trash:
            cb = self.trash.pop()
            cb()


def _build_reverse_device_attrs():
    prefix = "CU_DEVICE_ATTRIBUTE_"
    map = utils.UniqueDict()
    for name in dir(enums):
        if name.startswith(prefix):
            map[name[len(prefix):]] = getattr(enums, name)
    return map


DEVICE_ATTRIBUTES = _build_reverse_device_attrs()


class Device(object):
    """
    The device object owns the CUDA contexts.  This is owned by the driver
    object.  User should not construct devices directly.
    """
    def __init__(self, devnum):
        got_devnum = c_int()
        driver.cuDeviceGet(byref(got_devnum), devnum)
        assert devnum == got_devnum.value, "Driver returned another device"
        self.id = got_devnum.value
        self.trashing = TrashService("cuda.device%d.trash" % self.id)
        self.attributes = {}
        # Read compute capability
        cc_major = c_int()
        cc_minor = c_int()
        driver.cuDeviceComputeCapability(byref(cc_major), byref(cc_minor),
                                         self.id)
        self.compute_capability = (cc_major.value, cc_minor.value)
        # Read name
        bufsz = 128
        buf = (c_char * bufsz)()
        driver.cuDeviceGetName(buf, bufsz, self.id)
        self.name = buf.value
        # A dictionary or all context with handle value as the key
        self.contexts = {}

    def __del__(self):
        try:
            self.reset()
        except:
            traceback.print_exc()

    def __repr__(self):
        return "<CUDA device %d '%s'>" % (self.id, self.name)

    def __getattr__(self, attr):
        """Read attributes lazily
        """
        try:
            code = DEVICE_ATTRIBUTES[attr]
        except KeyError:
            raise AttributeError(attr)

        value = c_int()
        driver.cuDeviceGetAttribute(byref(value), code, self.id)
        setattr(self, attr, value)

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        if isinstance(other, Device):
            return self.id == other.id
        return False

    def __ne__(self, other):
        return not (self == other)

    def create_context(self):
        met_requirement_for_device(self)

        flags = 0
        if self.CAN_MAP_HOST_MEMORY:
            flags |= enums.CU_CTX_MAP_HOST

        # Clean up any trash
        self.trashing.service()

        # Create new context
        handle = drvapi.cu_context()
        driver.cuCtxCreate(byref(handle), flags, self.id)

        ctx = Context(weakref.proxy(self), handle,
                      _context_finalizer(self.trashing, handle))
        self.contexts[handle.value] = ctx
        return weakref.proxy(ctx)

    def close_all_context(self):
        # for ctx in self.contexts.values():
        #     ctx.reset()
        self.contexts.clear()

    def get_context(self):
        handle = drvapi.cu_context()
        self.cuCtxGetCurrent(byref(handle))
        if not handle.value:
            return None
        try:
            return self.contexts[handle.value]
        except KeyError:
            raise RuntimeError("Current context is not manged")

    def get_or_create_context(self):
        ctx = self.get_context()
        if ctx is None:
            ctx = self.create_context()
        return ctx

    def reset(self):
        self.close_all_context()
        self.trashing.clear()


def _context_finalizer(trashing, ctxhandle):
    def core():
        trashing.add_trash(lambda: driver.cuCtxDestroy(ctxhandle))
    return core


def met_requirement_for_device(device):
    if device.compute_capability < MIN_REQUIRED_CC:
        raise CudaSupportError("%s has compute capability < %s" %
                               (device, MIN_REQUIRED_CC))


class Context(object):
    """This object is tied to the lifetime of the actual context resource.

    This object is usually wrapped in a weakref proxy for user.  User seldom
    owns this object.

    """
    def __init__(self, device, handle, finalizer=None):
        self.device = device
        self.handle = handle
        self.finalizer = finalizer
        self.trashing = TrashService("cuda.device%d.context%x.trash" %
                                     (self.device.id, self.handle.value))
        self.is_managed = finalizer is not None
        self.allocations = utils.UniqueDict()
        self.modules = utils.UniqueDict()

    def __del__(self):
        try:
            self.reset()
            # Free itself
            if self.is_managed:
                self.finalizer()
        except:
            traceback.print_exc()

    def reset(self):
        """Clean up all owned resources in this context
        """
        # Free owned resources
        self.allocations.clear()
        self.modules.clear()
        # Clear trash
        self.trashing.clear()

    def get_memory_info(self):
        """Returns (free, total) memory in bytes in the context.
        """
        free = c_size_t()
        total = c_size_t()
        driver.cuMemGetInfo(byref(free), byref(total))
        return free.value, total.value

    def push(self):
        """Push context
        """
        driver.cuCtxPushCurrent(self.handle)

    def pop(self):
        """Pop context
        """
        driver.cuCtxPopCurrent(self.handle)

    def memalloc(self, bytesize):
        self.trashing.service()
        ptr = drvapi.cu_device_ptr()
        driver.cuMemAlloc(byref(ptr), bytesize)
        mem = MemoryPointer(weakref.proxy(self), ptr, bytesize,
                            _memory_finalizer(self.trashing, ptr))
        self.allocations[ptr.value] = mem
        return mem.own()

    def memfree(self, pointer):
        try:
            del self.allocations[pointer.value]
        except KeyError:
            raise RuntimeError("Freeing unmanaged device memory")
        self.trashing.service()

    def create_module_ptx(self, ptx):
        image = c_char_p(ptx)
        return self.create_module_image(image)

    def create_module_image(self, image):
        self.trashing.service()
        module = load_module_image(self, image)
        self.modules[module.handle.value] = module
        return weakref.proxy(module)

    def unload_module(self, module):
        del self.modules[module.handle.value]

    def create_stream(self):
        self.trashing.service()
        handle = drvapi.cu_stream()
        driver.cuStreamCreate(byref(handle), 0)
        return Stream(weakref.proxy(self), handle,
                      _stream_finalizer(self.trashing, handle))

    def __repr__(self):
        return "<CUDA context %s of device %d>" % (self.handle, self.device.id)


def load_module_image(context, image):
    """
    image must be a pointer
    """
    logsz = os.environ.get('NUMBAPRO_CUDA_LOG_SIZE', 1024)

    jitinfo = (c_char * logsz)()
    jiterrors = (c_char * logsz)()

    options = {
        enums.CU_JIT_INFO_LOG_BUFFER              : addressof(jitinfo),
        enums.CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES   : c_void_p(logsz),
        enums.CU_JIT_ERROR_LOG_BUFFER             : addressof(jiterrors),
        enums.CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES  : c_void_p(logsz),
        enums.CU_JIT_LOG_VERBOSE                  : c_void_p(VERBOSE_JIT_LOG),
    }

    option_keys = (drvapi.cu_jit_option * len(options))(*options.keys())
    option_vals = (c_void_p * len(options))(*options.values())

    handle = drvapi.cu_module()
    driver.cuModuleLoadDataEx(byref(handle), image, len(options),
                              option_keys, option_vals)

    info_log = jitinfo.value

    return Module(weakref.proxy(context), handle, info_log,
                  _module_finalizer(context.trashing), handle)


def _stream_finalizer(trashing, handle):
    def core():
        trashing.add_trash(lambda: driver.cuStreamDestroy(handle))
    return core


def _module_finalizer(trashing, handle):
    def core():
        trashing.add_trash(lambda: driver.cuModuleUnload(handle))
    return core


def _memory_finalizer(trashing, ptr):
    def core():
        trashing.add_trash(lambda: driver.cuMemFree(ptr))
    return core


class MemoryPointer(object):
    def __init__(self, context, pointer, size, finalizer=None):
        self.context = context
        self.pointer = pointer
        self.size = size
        self.finalizer = finalizer
        self.is_managed = finalizer is not None
        self.refct = 0

    def __del__(self):
        try:
            if self.is_managed:
                self.finalizer()
        except:
            traceback.print_exc()

    def own(self):
        return OwnedPointer(weakref.proxy(self))

    def free(self):
        """
        Forces the device memory to the trash.
        """
        self.context.memfree(self.pointer)

    def memset(self, byte, count=None, stream=0):
        count = self.size if count is None else count
        if stream:
            driver.cuMemsetD8Async(self.pointer, byte, count, stream.handle)
        else:
            driver.cuMemsetD8(self.pointer, byte, count)

    def view(self, start, stop=None):
        base = self.pointer.value + start
        if stop is None:
            size = self.size - start
        else:
            size = stop - start
        assert size > 0, "zero or negative memory size"
        pointer = drvapi.cu_device_ptr(base)
        view = MemoryPointer(self.context, pointer, size)
        return OwnedPointer(weakref.proxy(self), view)


class OwnedPointer(object):
    def __init__(self, memptr, view=None):
        self._mem = memptr
        self._mem.refct += 1
        if view is None:
            self._view = self._mem
        else:
            assert not view.is_managed
            self._view = view

    def __del__(self):
        try:
            self._mem.refct -= 1
            assert self._mem.refct >= 0
            if self._mem.refct == 0:
                self._mem.free()
        except weakref.ReferenceError:
            pass
        except:
            traceback.print_exc()

    def __getattr__(self, fname):
        """Proxy MemoryPointer methods
        """
        return getattr(self._view, fname)


class Stream(object):
    def __init__(self, context, handle, finalizer):
        self.context = context
        self.handle = handle
        self.finalizer = finalizer
        self.is_managed = finalizer is not None

    def __del__(self):
        try:
            if self.is_managed:
                self.finalizer()
        except:
            traceback.print_exc()

    def __int__(self):
        return self.handle.value

    def __repr__(self):
        return "<CUDA stream %d on %s>" % (self.handle.value, self.context)

    def synchronize(self):
        driver.cuStreamSynchronize(self.handle)

    @contextlib.contextmanager
    def auto_synchronize(self):
        yield self
        self.synchronize()


class Module(object):
    def __init__(self, context, handle, info_log, finalizer=None):
        self.context = context
        self.handle = handle
        self.info_log = info_log
        self.finalizer = finalizer
        self.is_managed = self.finalizer is not None

    def __del__(self):
        try:
            if self.is_managed:
                self.finalizer()
        except:
            traceback.print_exc()

    def unload(self):
        self.context.unload_module(self)
