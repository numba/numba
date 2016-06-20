"""
CUDA driver bridge implementation

NOTE:
The new driver implementation uses a "trashing service" that help prevents a
crashing the system (particularly OSX) when the CUDA context is corrupted at
resource deallocation.  The old approach ties resource management directly
into the object destructor; thus, at corruption of the CUDA context,
subsequent deallocation could further corrupt the CUDA context and causes the
system to freeze in some cases.

"""

from __future__ import absolute_import, print_function, division
import sys
import os
import ctypes
import weakref
import functools
import copy
import warnings
from ctypes import (c_int, byref, c_size_t, c_char, c_char_p, addressof,
                    c_void_p, c_float)
import contextlib
import numpy as np
from collections import namedtuple

from numba import utils, servicelib, mviewbuf
from .error import CudaSupportError, CudaDriverError
from .drvapi import API_PROTOTYPES
from .drvapi import cu_occupancy_b2d_size
from . import enums, drvapi
from numba import config
from numba.utils import longint as long

VERBOSE_JIT_LOG = int(os.environ.get('NUMBAPRO_VERBOSE_CU_JIT_LOG', 1))
MIN_REQUIRED_CC = (2, 0)


class DeadMemoryError(RuntimeError):
    pass


class LinkerError(RuntimeError):
    pass


class CudaAPIError(CudaDriverError):
    def __init__(self, code, msg):
        self.code = code
        self.msg = msg
        super(CudaAPIError, self).__init__(code, msg)

    def __str__(self):
        return "[%s] %s" % (self.code, self.msg)


def find_driver():
    envpath = os.environ.get('NUMBAPRO_CUDA_DRIVER', None)
    if envpath == '0':
        # Force fail
        _raise_driver_not_found()

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


def _getpid():
    return os.getpid()


ERROR_MAP = _build_reverse_error_map()

MISSING_FUNCTION_ERRMSG = """driver missing function: %s.
Requires CUDA 5.5 or above.
"""


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
            cls._singleton = obj
        return obj

    def __init__(self):
        self.devices = utils.UniqueDict()
        self.is_initialized = False
        self.initialization_error = None
        self.pid = None
        try:
            if config.DISABLE_CUDA:
                raise CudaSupportError("CUDA disabled by user")
            self.lib = find_driver()
        except CudaSupportError as e:
            self.is_initialized = True
            self.initialization_error = e

    def initialize(self):
        self.is_initialized = True
        try:
            self.cuInit(0)
        except CudaAPIError as e:
            self.initialization_error = e
            raise CudaSupportError("Error at driver init: \n%s:" % e)
        else:
            self.pid = _getpid()

    @property
    def is_available(self):
        if not self.is_initialized:
            self.initialize()
        return self.initialization_error is None

    def __getattr__(self, fname):
        # First request of a driver API function
        try:
            proto = API_PROTOTYPES[fname]
        except KeyError:
            raise AttributeError(fname)
        restype = proto[0]
        argtypes = proto[1:]

        # Initialize driver
        if not self.is_initialized:
            self.initialize()

        if self.initialization_error is not None:
            raise CudaSupportError("Error at driver init: \n%s:" %
                                   self.initialization_error)

        # Find function in driver library
        libfn = self._find_api(fname)
        libfn.restype = restype
        libfn.argtypes = argtypes

        @functools.wraps(libfn)
        def safe_cuda_api_call(*args):
            retcode = libfn(*args)
            self._check_error(fname, retcode)

        setattr(self, fname, safe_cuda_api_call)
        return safe_cuda_api_call

    def _find_api(self, fname):
        # Try version 2
        try:
            return getattr(self.lib, fname + "_v2")
        except AttributeError:
            pass

        # Try regular
        try:
            return getattr(self.lib, fname)
        except AttributeError:
            pass

        # Not found.
        # Delay missing function error to use
        def absent_function(*args, **kws):
            raise CudaDriverError(MISSING_FUNCTION_ERRMSG % fname)

        setattr(self, fname, absent_function)
        return absent_function

    def _check_error(self, fname, retcode):
        if retcode != enums.CUDA_SUCCESS:
            errname = ERROR_MAP.get(retcode, "UNKNOWN_CUDA_ERROR")
            msg = "Call to %s results in %s" % (fname, errname)
            raise CudaAPIError(retcode, msg)

    def get_device(self, devnum=0):
        dev = self.devices.get(devnum)
        if dev is None:
            dev = Device(devnum)
            self.devices[devnum] = dev
        return weakref.proxy(dev)

    def get_device_count(self):
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

    def get_context(self):
        """Get current active context in CUDA driver runtime.
        Note: Lowlevel calls that returns the handle.
        """
        # Detect forking
        if self.is_initialized and _getpid() != self.pid:
            raise CudaDriverError("CUDA initialized before forking")

        handle = drvapi.cu_context(0)
        driver.cuCtxGetCurrent(byref(handle))
        if not handle.value:
            return None
        return handle

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

    @contextlib.contextmanager
    def defer_cleanup(self):
        orig = self.enabled
        self.enabled = False
        yield
        self.enabled = orig
        self.service()


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
        self.trashing = trashing = TrashService("cuda.device%d.trash" % self.id)
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

        def finalizer():
            trashing.clear()

        utils.finalize(self, finalizer)

    @property
    def COMPUTE_CAPABILITY(self):
        """
        For backward compatibility
        """
        warnings.warn("Deprecated attribute 'COMPUTE_CAPABILITY'; use lower "
                      "case version", DeprecationWarning)
        return self.compute_capability

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
        setattr(self, attr, value.value)

        return value.value

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        if isinstance(other, Device):
            return self.id == other.id
        return False

    def __ne__(self, other):
        return not (self == other)

    def create_context(self):
        """Create a CUDA context.
        """
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

        return ctx

    def reset(self):
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
    """
    This object wraps a CUDA Context resource.

    Contexts should not be constructed directly by user code.
    """

    def __init__(self, device, handle, finalizer=None):
        self.device = device
        self.handle = handle
        self.external_finalizer = finalizer
        self.trashing = TrashService("cuda.device%d.context%x.trash" %
                                     (self.device.id, self.handle.value))
        self.allocations = utils.UniqueDict()
        self.modules = utils.UniqueDict()
        utils.finalize(self, self._make_finalizer())
        # For storing context specific data
        self.extras = {}

    def _make_finalizer(self):
        """
        Make a finalizer function that doesn't keep a reference to this object.
        """
        allocations = self.allocations
        modules = self.modules
        trashing = self.trashing
        external_finalizer = self.external_finalizer

        def finalize():
            allocations.clear()
            modules.clear()
            trashing.clear()
            if external_finalizer is not None:
                external_finalizer()
        return finalize

    def reset(self):
        """
        Clean up all owned resources in this context.
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

    def get_active_blocks_per_multiprocessor(self, func, blocksize, memsize, flags=None):
        """Return occupancy of a function.
        :param func: kernel for which occupancy is calculated
        :param blocksize: block size the kernel is intended to be launched with
        :param memsize: per-block dynamic shared memory usage intended, in bytes"""

        retval = c_int()
        if not flags:
            driver.cuOccupancyMaxActiveBlocksPerMultiprocessor(byref(retval), func.handle, blocksize, memsize)
        else:
            driver.cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(byref(retval), func.handle, blocksize, memsize, flags)
        return retval.value

    def get_max_potential_block_size(self, func, b2d_func, memsize, blocksizelimit, flags=None):
        """Suggest a launch configuration with reasonable occupancy.
        :param func: kernel for which occupancy is calculated
        :param b2d_func: function that calculates how much per-block dynamic shared memory 'func'
          uses based on the block size.
        :param memsize: per-block dynamic shared memory usage intended, in bytes
        :param blocksizelimit: maximum block size the kernel is designed to handle"""

        gridsize = c_int()
        blocksize = c_int()
        b2d_cb = cu_occupancy_b2d_size(b2d_func)
        if not flags:
            driver.cuOccupancyMaxPotentialBlockSize(byref(gridsize), byref(blocksize),
                                                    func.handle,
                                                    b2d_cb,
                                                    memsize, blocksizelimit)
        else:
            driver.cuOccupancyMaxPotentialBlockSizeWithFlags(byref(gridsize), byref(blocksize),
                                                             func.handle, b2d_cb,
                                                             memsize, blocksizelimit, flags)
        return (gridsize.value, blocksize.value)

    def push(self):
        """
        Pushes this context on the current CPU Thread.
        """
        driver.cuCtxPushCurrent(self.handle)

    def pop(self):
        """
        Pops this context off the current CPU thread. Note that this context must
        be at the top of the context stack, otherwise an error will occur.
        """
        popped = drvapi.cu_context()
        driver.cuCtxPopCurrent(byref(popped))
        assert popped.value == self.handle.value

    def memalloc(self, bytesize):
        self.trashing.service()
        ptr = drvapi.cu_device_ptr()
        driver.cuMemAlloc(byref(ptr), bytesize)
        _memory_finalizer = _make_mem_finalizer(driver.cuMemFree)
        mem = MemoryPointer(weakref.proxy(self), ptr, bytesize,
                            _memory_finalizer(self, ptr))
        self.allocations[ptr.value] = mem
        return mem.own()

    def memhostalloc(self, bytesize, mapped=False, portable=False, wc=False):
        self.trashing.service()

        pointer = c_void_p()
        flags = 0
        if mapped:
            flags |= enums.CU_MEMHOSTALLOC_DEVICEMAP
        if portable:
            flags |= enums.CU_MEMHOSTALLOC_PORTABLE
        if wc:
            flags |= enums.CU_MEMHOSTALLOC_WRITECOMBINED

        driver.cuMemHostAlloc(byref(pointer), bytesize, flags)
        owner = None

        if mapped:
            _hostalloc_finalizer = _make_mem_finalizer(driver.cuMemFreeHost)
            finalizer = _hostalloc_finalizer(self, pointer)
            mem = MappedMemory(weakref.proxy(self), owner, pointer,
                               bytesize, finalizer=finalizer)

            self.allocations[mem.handle.value] = mem
            return mem.own()
        else:
            finalizer = _pinnedalloc_finalizer(self.trashing, pointer)
            mem = PinnedMemory(weakref.proxy(self), owner, pointer, bytesize,
                               finalizer=finalizer)
            return mem

    def mempin(self, owner, pointer, size, mapped=False):
        self.trashing.service()

        if isinstance(pointer, (int, long)):
            pointer = c_void_p(pointer)

        if mapped and not self.device.CAN_MAP_HOST_MEMORY:
            raise CudaDriverError("%s cannot map host memory" % self.device)

        # possible flags are "portable" (between context)
        # and "device-map" (map host memory to device thus no need
        # for memory transfer).
        flags = 0

        if mapped:
            flags |= enums.CU_MEMHOSTREGISTER_DEVICEMAP

        driver.cuMemHostRegister(pointer, size, flags)

        if mapped:
            _mapped_finalizer = _make_mem_finalizer(driver.cuMemHostUnregister)
            finalizer = _mapped_finalizer(self, pointer)
            mem = MappedMemory(weakref.proxy(self), owner, pointer, size,
                               finalizer=finalizer)
            self.allocations[mem.handle.value] = mem
            return mem.own()
        else:
            mem = PinnedMemory(weakref.proxy(self), owner, pointer, size,
                               finalizer=_pinned_finalizer(self.trashing,
                                                           pointer))
            return mem

    def memunpin(self, pointer):
        raise NotImplementedError

    def create_module_ptx(self, ptx):
        if isinstance(ptx, str):
            ptx = ptx.encode('utf8')
        image = c_char_p(ptx)
        return self.create_module_image(image)

    def create_module_image(self, image):
        self.trashing.service()
        module = load_module_image(self, image)
        self.modules[module.handle.value] = module
        return weakref.proxy(module)

    def unload_module(self, module):
        del self.modules[module.handle.value]
        self.trashing.service()

    def create_stream(self):
        self.trashing.service()
        handle = drvapi.cu_stream()
        driver.cuStreamCreate(byref(handle), 0)
        return Stream(weakref.proxy(self), handle,
                      _stream_finalizer(self.trashing, handle))

    def create_event(self, timing=True):
        self.trashing.service()

        handle = drvapi.cu_event()
        flags = 0
        if not timing:
            flags |= enums.CU_EVENT_DISABLE_TIMING
        driver.cuEventCreate(byref(handle), flags)
        return Event(weakref.proxy(self), handle,
                     finalizer=_event_finalizer(self.trashing, handle))

    def synchronize(self):
        driver.cuCtxSynchronize()

    def __repr__(self):
        return "<CUDA context %s of device %d>" % (self.handle, self.device.id)

    def __eq__(self, other):
        if isinstance(other, Context):
            return self.handle == other.handle
        else:
            return NotImplemented

    def __ne__(self, other):
        return not self.__eq__(other)


def load_module_image(context, image):
    """
    image must be a pointer
    """
    logsz = os.environ.get('NUMBAPRO_CUDA_LOG_SIZE', 1024)

    jitinfo = (c_char * logsz)()
    jiterrors = (c_char * logsz)()

    options = {
        enums.CU_JIT_INFO_LOG_BUFFER: addressof(jitinfo),
        enums.CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES: c_void_p(logsz),
        enums.CU_JIT_ERROR_LOG_BUFFER: addressof(jiterrors),
        enums.CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES: c_void_p(logsz),
        enums.CU_JIT_LOG_VERBOSE: c_void_p(VERBOSE_JIT_LOG),
    }

    option_keys = (drvapi.cu_jit_option * len(options))(*options.keys())
    option_vals = (c_void_p * len(options))(*options.values())

    handle = drvapi.cu_module()
    try:
        driver.cuModuleLoadDataEx(byref(handle), image, len(options),
                                  option_keys, option_vals)
    except CudaAPIError as e:
        msg = "cuModuleLoadDataEx error:\n%s" % jiterrors.value.decode("utf8")
        raise CudaAPIError(e.code, msg)

    info_log = jitinfo.value

    return Module(weakref.proxy(context), handle, info_log,
                  _module_finalizer(context, handle))


def _make_mem_finalizer(dtor):
    def mem_finalize(context, handle):
        trashing = context.trashing
        allocations = context.allocations

        def core():
            def cleanup():
                if allocations:
                    del allocations[handle.value]
                dtor(handle)

            trashing.add_trash(cleanup)

        return core

    return mem_finalize


def _pinnedalloc_finalizer(trashing, handle):
    def core():
        trashing.add_trash(lambda: driver.cuMemFreeHost(handle))

    return core


def _pinned_finalizer(trashing, handle):
    def core():
        trashing.add_trash(lambda: driver.cuMemHostUnregister(handle))

    return core


def _event_finalizer(trashing, handle):
    def core():
        trashing.add_trash(lambda: driver.cuEventDestroy(handle))

    return core


def _stream_finalizer(trashing, handle):
    def core():
        trashing.add_trash(lambda: driver.cuStreamDestroy(handle))

    return core


def _module_finalizer(context, handle):
    trashing = context.trashing
    modules = context.modules

    def core():
        def cleanup():
            # All modules are owned by their parent Context.
            # A Module is either released by a call to
            # Context.unload_module, which clear the handle (pointer) mapping
            # (checked by the following assertion), or, by Context.reset().
            # Both releases the sole reference to the Module and trigger the
            # finalizer for the Module instance.  The actual call to
            # cuModuleUnload is deferred to the trashing service to avoid
            # further corruption of the CUDA context if a fatal error has
            # occurred in the CUDA driver.
            assert handle.value not in modules
            driver.cuModuleUnload(handle)

        trashing.add_trash(cleanup)

    return core


class MemoryPointer(object):
    __cuda_memory__ = True

    def __init__(self, context, pointer, size, finalizer=None, owner=None):
        self.context = context
        self.device_pointer = pointer
        self.size = size
        self._cuda_memsize_ = size
        self.is_managed = finalizer is not None
        self.refct = 0
        self.handle = self.device_pointer
        self._owner = owner

        if finalizer is not None:
            self._finalizer = utils.finalize(self, finalizer)

    @property
    def owner(self):
        return self if self._owner is None else self._owner

    def own(self):
        return OwnedPointer(weakref.proxy(self))

    def free(self):
        """
        Forces the device memory to the trash.
        """
        if self.is_managed:
            if not self._finalizer.alive:
                raise RuntimeError("Freeing dead memory")
            self._finalizer()
            assert not self._finalizer.alive

    def memset(self, byte, count=None, stream=0):
        count = self.size if count is None else count
        if stream:
            driver.cuMemsetD8Async(self.device_pointer, byte, count,
                                   stream.handle)
        else:
            driver.cuMemsetD8(self.device_pointer, byte, count)

    def view(self, start, stop=None):
        base = self.device_pointer.value + start
        if stop is None:
            size = self.size - start
        else:
            size = stop - start
        assert size > 0, "zero or negative memory size"
        pointer = drvapi.cu_device_ptr(base)
        view = MemoryPointer(self.context, pointer, size, owner=self.owner)
        return OwnedPointer(weakref.proxy(self.owner), view)

    @property
    def device_ctypes_pointer(self):
        return self.device_pointer


class MappedMemory(MemoryPointer):
    __cuda_memory__ = True

    def __init__(self, context, owner, hostpointer, size,
                 finalizer=None):
        self.owned = owner
        self.host_pointer = hostpointer
        devptr = drvapi.cu_device_ptr()
        driver.cuMemHostGetDevicePointer(byref(devptr), hostpointer, 0)
        self.device_pointer = devptr
        super(MappedMemory, self).__init__(context, devptr, size,
                                           finalizer=finalizer)
        self.handle = self.host_pointer

        # For buffer interface
        self._buflen_ = self.size
        self._bufptr_ = self.host_pointer.value

    def own(self):
        return MappedOwnedPointer(weakref.proxy(self))


class PinnedMemory(mviewbuf.MemAlloc):
    def __init__(self, context, owner, pointer, size, finalizer=None):
        self.context = context
        self.owned = owner
        self.size = size
        self.host_pointer = pointer
        self.is_managed = finalizer is not None
        self.handle = self.host_pointer

        # For buffer interface
        self._buflen_ = self.size
        self._bufptr_ = self.host_pointer.value

        if finalizer is not None:
            utils.finalize(self, finalizer)

    def own(self):
        return self


class OwnedPointer(object):
    def __init__(self, memptr, view=None):
        self._mem = memptr

        if view is None:
            self._view = self._mem
        else:
            assert not view.is_managed
            self._view = view

        mem = self._mem

        def deref():
            mem.refct -= 1
            assert mem.refct >= 0
            if mem.refct == 0:
                mem.free()

        self._mem.refct += 1
        utils.finalize(self, deref)

    def __getattr__(self, fname):
        """Proxy MemoryPointer methods
        """
        return getattr(self._view, fname)


class MappedOwnedPointer(OwnedPointer, mviewbuf.MemAlloc):
    pass


class Stream(object):
    def __init__(self, context, handle, finalizer):
        self.context = context
        self.handle = handle
        if finalizer is not None:
            utils.finalize(self, finalizer)

    def __int__(self):
        return self.handle.value

    def __repr__(self):
        return "<CUDA stream %d on %s>" % (self.handle.value, self.context)

    def synchronize(self):
        '''
        Wait for all commands in this stream to execute. This will commit any
        pending memory transfers.
        '''
        driver.cuStreamSynchronize(self.handle)

    @contextlib.contextmanager
    def auto_synchronize(self):
        '''
        A context manager that waits for all commands in this stream to execute
        and commits any pending memory transfers upon exiting the context.
        '''
        yield self
        self.synchronize()


class Event(object):
    def __init__(self, context, handle, finalizer=None):
        self.context = context
        self.handle = handle
        if finalizer is not None:
            utils.finalize(self, finalizer)

    def query(self):
        """
        Returns True if all work before the most recent record has completed;
        otherwise, returns False.
        """
        try:
            driver.cuEventQuery(self.handle)
        except CudaAPIError as e:
            if e.code == enums.CUDA_ERROR_NOT_READY:
                return False
            else:
                raise
        else:
            return True

    def record(self, stream=0):
        """
        Set the record point of the event to the current point in the given
        stream.

        The event will be considered to have occurred when all work that was
        queued in the stream at the time of the call to ``record()`` has been
        completed.
        """
        hstream = stream.handle if stream else 0
        driver.cuEventRecord(self.handle, hstream)

    def synchronize(self):
        """
        Synchronize the host thread for the completion of the event.
        """
        driver.cuEventSynchronize(self.handle)

    def wait(self, stream=0):
        """
        All future works submitted to stream will wait util the event completes.
        """
        hstream = stream.handle if stream else 0
        flags = 0
        driver.cuStreamWaitEvent(hstream, self.handle, flags)

    def elapsed_time(self, evtend):
        return event_elapsed_time(self, evtend)


def event_elapsed_time(evtstart, evtend):
    '''
    Compute the elapsed time between two events in milliseconds.
    '''
    msec = c_float()
    driver.cuEventElapsedTime(byref(msec), evtstart.handle, evtend.handle)
    return msec.value


class Module(object):
    def __init__(self, context, handle, info_log, finalizer=None):
        self.context = context
        self.handle = handle
        self.info_log = info_log
        if finalizer is not None:
            self._finalizer = utils.finalize(self, finalizer)

    def unload(self):
        self.context.unload_module(self)

    def get_function(self, name):
        handle = drvapi.cu_function()
        driver.cuModuleGetFunction(byref(handle), self.handle,
                                   name.encode('utf8'))
        return Function(weakref.proxy(self), handle, name)

    def get_global_symbol(self, name):
        ptr = drvapi.cu_device_ptr()
        size = drvapi.c_size_t()
        driver.cuModuleGetGlobal(byref(ptr), byref(size), self.handle,
                                 name.encode('utf8'))
        return MemoryPointer(self.context, ptr, size), size.value


FuncAttr = namedtuple("FuncAttr", ["regs", "shared", "local", "const",
                                   "maxthreads"])


class Function(object):
    griddim = 1, 1, 1
    blockdim = 1, 1, 1
    stream = 0
    sharedmem = 0

    def __init__(self, module, handle, name):
        self.module = module
        self.handle = handle
        self.name = name
        self.attrs = self._read_func_attr_all()

    def __repr__(self):
        return "<CUDA function %s>" % self.name

    def cache_config(self, prefer_equal=False, prefer_cache=False,
                     prefer_shared=False):
        prefer_equal = prefer_equal or (prefer_cache and prefer_shared)
        if prefer_equal:
            flag = enums.CU_FUNC_CACHE_PREFER_EQUAL
        elif prefer_cache:
            flag = enums.CU_FUNC_CACHE_PREFER_L1
        elif prefer_shared:
            flag = enums.CU_FUNC_CACHE_PREFER_SHARED
        else:
            flag = enums.CU_FUNC_CACHE_PREFER_NONE
        driver.cuFuncSetCacheConfig(self.handle, flag)

    def configure(self, griddim, blockdim, sharedmem=0, stream=0):
        while len(griddim) < 3:
            griddim += (1,)

        while len(blockdim) < 3:
            blockdim += (1,)

        inst = copy.copy(self)  # shallow clone the object
        inst.griddim = griddim
        inst.blockdim = blockdim
        inst.sharedmem = sharedmem
        if stream:
            inst.stream = stream
        else:
            inst.stream = 0
        return inst

    def __call__(self, *args):
        '''
        *args -- Must be either ctype objects of DevicePointer instances.
        '''
        if self.stream:
            streamhandle = self.stream.handle
        else:
            streamhandle = None

        launch_kernel(self.handle, self.griddim, self.blockdim,
                      self.sharedmem, streamhandle, args)

    @property
    def device(self):
        return self.module.context.device

    def _read_func_attr(self, attrid):
        """
        Read CUfunction attributes
        """
        retval = c_int()
        driver.cuFuncGetAttribute(byref(retval), attrid, self.handle)
        return retval.value

    def _read_func_attr_all(self):
        nregs = self._read_func_attr(enums.CU_FUNC_ATTRIBUTE_NUM_REGS)
        cmem = self._read_func_attr(enums.CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES)
        lmem = self._read_func_attr(enums.CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES)
        smem = self._read_func_attr(enums.CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES)
        maxtpb = self._read_func_attr(
            enums.CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK)
        return FuncAttr(regs=nregs, const=cmem, local=lmem, shared=smem,
                        maxthreads=maxtpb)


def launch_kernel(cufunc_handle, griddim, blockdim, sharedmem, hstream, args):
    gx, gy, gz = griddim
    bx, by, bz = blockdim

    param_vals = []
    for arg in args:
        if is_device_memory(arg):
            param_vals.append(addressof(device_ctypes_pointer(arg)))
        else:
            param_vals.append(addressof(arg))

    params = (c_void_p * len(param_vals))(*param_vals)

    driver.cuLaunchKernel(cufunc_handle,
                          gx, gy, gz,
                          bx, by, bz,
                          sharedmem,
                          hstream,
                          params,
                          None)


FILE_EXTENSION_MAP = {
    'o': enums.CU_JIT_INPUT_OBJECT,
    'ptx': enums.CU_JIT_INPUT_PTX,
    'a': enums.CU_JIT_INPUT_LIBRARY,
    'cubin': enums.CU_JIT_INPUT_CUBIN,
    'fatbin': enums.CU_JIT_INPUT_FATBINAR,
}


class Linker(object):
    def __init__(self):
        logsz = int(os.environ.get('NUMBAPRO_CUDA_LOG_SIZE', 1024))
        linkerinfo = (c_char * logsz)()
        linkererrors = (c_char * logsz)()

        options = {
            enums.CU_JIT_INFO_LOG_BUFFER: addressof(linkerinfo),
            enums.CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES: c_void_p(logsz),
            enums.CU_JIT_ERROR_LOG_BUFFER: addressof(linkererrors),
            enums.CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES: c_void_p(logsz),
            enums.CU_JIT_LOG_VERBOSE: c_void_p(1),
        }

        raw_keys = list(options.keys()) + [enums.CU_JIT_TARGET_FROM_CUCONTEXT]
        raw_values = list(options.values())
        del options

        option_keys = (drvapi.cu_jit_option * len(raw_keys))(*raw_keys)
        option_vals = (c_void_p * len(raw_values))(*raw_values)

        self.handle = handle = drvapi.cu_link_state()
        driver.cuLinkCreate(len(raw_keys), option_keys, option_vals,
                            byref(self.handle))

        utils.finalize(self, driver.cuLinkDestroy, handle)

        self.linker_info_buf = linkerinfo
        self.linker_errors_buf = linkererrors

        self._keep_alive = [linkerinfo, linkererrors, option_keys, option_vals]

    @property
    def info_log(self):
        return self.linker_info_buf.value.decode('utf8')

    @property
    def error_log(self):
        return self.linker_errors_buf.value.decode('utf8')

    def add_ptx(self, ptx, name='<cudapy-ptx>'):
        ptxbuf = c_char_p(ptx)
        namebuf = c_char_p(name.encode('utf8'))
        self._keep_alive += [ptxbuf, namebuf]
        try:
            driver.cuLinkAddData(self.handle, enums.CU_JIT_INPUT_PTX,
                                 ptxbuf, len(ptx), namebuf, 0, None, None)
        except CudaAPIError as e:
            raise LinkerError("%s\n%s" % (e, self.error_log))

    def add_file(self, path, kind):
        pathbuf = c_char_p(path.encode("utf8"))
        self._keep_alive.append(pathbuf)

        try:
            driver.cuLinkAddFile(self.handle, kind, pathbuf, 0, None, None)
        except CudaAPIError as e:
            raise LinkerError("%s\n%s" % (e, self.error_log))

    def add_file_guess_ext(self, path):
        ext = path.rsplit('.', 1)[1]
        kind = FILE_EXTENSION_MAP[ext]
        self.add_file(path, kind)

    def complete(self):
        '''
        Returns (cubin, size)
            cubin is a pointer to a internal buffer of cubin owned
            by the linker; thus, it should be loaded before the linker
            is destroyed.
        '''
        cubin = c_void_p(0)
        size = c_size_t(0)

        try:
            driver.cuLinkComplete(self.handle, byref(cubin), byref(size))
        except CudaAPIError as e:
            raise LinkerError("%s\n%s" % (e, self.error_log))

        size = size.value
        assert size > 0, 'linker returned a zero sized cubin'
        del self._keep_alive[:]
        return cubin, size


# -----------------------------------------------------------------------------


def _device_pointer_attr(devmem, attr, odata):
    """Query attribute on the device pointer
    """
    error = driver.cuPointerGetAttribute(byref(odata), attr,
                                         device_ctypes_pointer(devmem))
    driver.check_error(error, "Failed to query pointer attribute")


def device_pointer_type(devmem):
    """Query the device pointer type: host, device, array, unified?
    """
    ptrtype = c_int(0)
    _device_pointer_attr(devmem, enums.CU_POINTER_ATTRIBUTE_MEMORY_TYPE,
                         ptrtype)
    map = {
        enums.CU_MEMORYTYPE_HOST: 'host',
        enums.CU_MEMORYTYPE_DEVICE: 'device',
        enums.CU_MEMORYTYPE_ARRAY: 'array',
        enums.CU_MEMORYTYPE_UNIFIED: 'unified',
    }
    return map[ptrtype.value]


def device_extents(devmem):
    """Find the extents (half open begin and end pointer) of the underlying
    device memory allocation.

    NOTE: it always returns the extents of the allocation but the extents
    of the device memory view that can be a subsection of the entire allocation.
    """
    s = drvapi.cu_device_ptr()
    n = c_size_t()
    devptr = device_ctypes_pointer(devmem)
    driver.cuMemGetAddressRange(byref(s), byref(n), devptr)
    s, n = s.value, n.value
    return s, s + n


def device_memory_size(devmem):
    """Check the memory size of the device memory.
    The result is cached in the device memory object.
    It may query the driver for the memory size of the device memory allocation.
    """
    sz = getattr(devmem, '_cuda_memsize_', None)
    if sz is None:
        s, e = device_extents(devmem)
        sz = e - s
        devmem._cuda_memsize_ = sz
    assert sz > 0, "zero length array"
    return sz


def host_pointer(obj):
    """
    NOTE: The underlying data pointer from the host data buffer is used and
    it should not be changed until the operation which can be asynchronous
    completes.
    """
    if isinstance(obj, (int, long)):
        return obj

    forcewritable = isinstance(obj, np.void)
    return mviewbuf.memoryview_get_buffer(obj, forcewritable)


def host_memory_extents(obj):
    "Returns (start, end) the start and end pointer of the array (half open)."
    return mviewbuf.memoryview_get_extents(obj)


def memory_size_from_info(shape, strides, itemsize):
    """et the byte size of a contiguous memory buffer given the shape, strides
    and itemsize.
    """
    assert len(shape) == len(strides), "# dim mismatch"
    ndim = len(shape)
    s, e = mviewbuf.memoryview_get_extents_info(shape, strides, ndim, itemsize)
    return e - s


def host_memory_size(obj):
    "Get the size of the memory"
    s, e = host_memory_extents(obj)
    assert e >= s, "memory extend of negative size"
    return e - s


def device_pointer(obj):
    "Get the device pointer as an integer"
    return device_ctypes_pointer(obj).value


def device_ctypes_pointer(obj):
    "Get the ctypes object for the device pointer"
    if obj is None:
        return c_void_p(0)
    require_device_memory(obj)
    return obj.device_ctypes_pointer


def is_device_memory(obj):
    """All CUDA memory object is recognized as an instance with the attribute
    "__cuda_memory__" defined and its value evaluated to True.

    All CUDA memory object should also define an attribute named
    "device_pointer" which value is an int(or long) object carrying the pointer
    value of the device memory address.  This is not tested in this method.
    """
    return getattr(obj, '__cuda_memory__', False)


def require_device_memory(obj):
    """A sentry for methods that accept CUDA memory object.
    """
    if not is_device_memory(obj):
        raise Exception("Not a CUDA memory object.")


def device_memory_depends(devmem, *objs):
    """Add dependencies to the device memory.

    Mainly used for creating structures that points to other device memory,
    so that the referees are not GC and released.
    """
    depset = getattr(devmem, "_depends_", [])
    depset.extend(objs)


def host_to_device(dst, src, size, stream=0):
    """
    NOTE: The underlying data pointer from the host data buffer is used and
    it should not be changed until the operation which can be asynchronous
    completes.
    """
    varargs = []

    if stream:
        assert isinstance(stream, Stream)
        fn = driver.cuMemcpyHtoDAsync
        varargs.append(stream.handle)
    else:
        fn = driver.cuMemcpyHtoD

    fn(device_pointer(dst), host_pointer(src), size, *varargs)


def device_to_host(dst, src, size, stream=0):
    """
    NOTE: The underlying data pointer from the host data buffer is used and
    it should not be changed until the operation which can be asynchronous
    completes.
    """
    varargs = []

    if stream:
        assert isinstance(stream, Stream)
        fn = driver.cuMemcpyDtoHAsync
        varargs.append(stream.handle)
    else:
        fn = driver.cuMemcpyDtoH

    fn(host_pointer(dst), device_pointer(src), size, *varargs)


def device_to_device(dst, src, size, stream=0):
    """
    NOTE: The underlying data pointer from the host data buffer is used and
    it should not be changed until the operation which can be asynchronous
    completes.
    """
    varargs = []

    if stream:
        assert isinstance(stream, Stream)
        fn = driver.cuMemcpyDtoDAsync
        varargs.append(stream.handle)
    else:
        fn = driver.cuMemcpyDtoD

    fn(device_pointer(dst), device_pointer(src), size, *varargs)


def device_memset(dst, val, size, stream=0):
    """Memset on the device.
    If stream is not zero, asynchronous mode is used.

    dst: device memory
    val: byte value to be written
    size: number of byte to be written
    stream: a CUDA stream
    """
    varargs = []

    if stream:
        assert isinstance(stream, Stream)
        fn = driver.cuMemsetD8Async
        varargs.append(stream.handle)
    else:
        fn = driver.cuMemsetD8

    fn(device_pointer(dst), val, size, *varargs)


def profile_start():
    '''
    Enable profile collection in the current context.
    '''
    driver.cuProfilerStart()


def profile_stop():
    '''
    Disable profile collection in the current context.
    '''
    driver.cuProfilerStop()


@contextlib.contextmanager
def profiling():
    """
    Context manager that enables profiling on entry and disables profiling on
    exit.
    """
    profile_start()
    yield
    profile_stop()
