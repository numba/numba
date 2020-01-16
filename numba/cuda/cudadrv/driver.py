"""
CUDA driver bridge implementation

NOTE:
The new driver implementation uses a *_PendingDeallocs* that help prevents a
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
import logging
import threading
from itertools import product
from ctypes import (c_int, byref, c_size_t, c_char, c_char_p, addressof,
                    c_void_p, c_float)
import contextlib
import numpy as np
from collections import namedtuple, deque

from numba import utils, mviewbuf
from .error import CudaSupportError, CudaDriverError
from .drvapi import API_PROTOTYPES
from .drvapi import cu_occupancy_b2d_size
from . import enums, drvapi, _extras
from numba import config, serialize, errors
from numba.utils import longint as long
from numba.cuda.envvars import get_numba_envvar


VERBOSE_JIT_LOG = int(get_numba_envvar('VERBOSE_CU_JIT_LOG', 1))
MIN_REQUIRED_CC = (2, 0)
SUPPORTS_IPC = sys.platform.startswith('linux')


def _make_logger():
    logger = logging.getLogger(__name__)
    # is logging configured?
    if not utils.logger_hasHandlers(logger):
        # read user config
        lvl = str(config.CUDA_LOG_LEVEL).upper()
        lvl = getattr(logging, lvl, None)
        if not isinstance(lvl, int):
            # default to critical level
            lvl = logging.CRITICAL
        logger.setLevel(lvl)
        # did user specify a level?
        if config.CUDA_LOG_LEVEL:
            # create a simple handler that prints to stderr
            handler = logging.StreamHandler(sys.stderr)
            fmt = '== CUDA [%(relativeCreated)d] %(levelname)5s -- %(message)s'
            handler.setFormatter(logging.Formatter(fmt=fmt))
            logger.addHandler(handler)
        else:
            # otherwise, put a null handler
            logger.addHandler(logging.NullHandler())
    return logger


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

    envpath = get_numba_envvar('CUDA_DRIVER')

    if envpath == '0':
        # Force fail
        _raise_driver_not_found()

    # Determine DLL type
    if sys.platform == 'win32':
        dlloader = ctypes.WinDLL
        dldir = ['\\windows\\system32']
        dlnames = ['nvcuda.dll']
    elif sys.platform == 'darwin':
        dlloader = ctypes.CDLL
        dldir = ['/usr/local/cuda/lib']
        dlnames = ['libcuda.dylib']
    else:
        # Assume to be *nix like
        dlloader = ctypes.CDLL
        dldir = ['/usr/lib', '/usr/lib64']
        dlnames = ['libcuda.so', 'libcuda.so.1']

    if envpath is not None:
        try:
            envpath = os.path.abspath(envpath)
        except ValueError:
            raise ValueError("NUMBA_CUDA_DRIVER %s is not a valid path" %
                             envpath)
        if not os.path.isfile(envpath):
            raise ValueError("NUMBA_CUDA_DRIVER %s is not a valid file "
                             "path.  Note it must be a filepath of the .so/"
                             ".dll/.dylib or the driver" % envpath)
        candidates = [envpath]
    else:
        # First search for the name in the default library path.
        # If that is not found, try the specific path.
        candidates = dlnames + [os.path.join(x, y)
                                for x, y in product(dldir, dlnames)]

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
try setting environment variable NUMBA_CUDA_DRIVER
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
Requires CUDA 8.0 or above.
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
                msg = ("CUDA is disabled due to setting NUMBA_DISABLE_CUDA=1 "
                       "in the environment, or because CUDA is unsupported on "
                       "32-bit systems.")
                raise CudaSupportError(msg)
            self.lib = find_driver()
        except CudaSupportError as e:
            self.is_initialized = True
            self.initialization_error = e

    def initialize(self):
        # lazily initialize logger
        global _logger
        _logger = _make_logger()

        self.is_initialized = True
        try:
            _logger.info('init')
            self.cuInit(0)
        except CudaAPIError as e:
            self.initialization_error = e
            raise CudaSupportError("Error at driver init: \n%s:" % e)
        else:
            self.pid = _getpid()

        self._initialize_extras()

    def _initialize_extras(self):
        # set pointer to original cuIpcOpenMemHandle
        set_proto = ctypes.CFUNCTYPE(None, c_void_p)
        set_cuIpcOpenMemHandle = set_proto(_extras.set_cuIpcOpenMemHandle)
        set_cuIpcOpenMemHandle(self._find_api('cuIpcOpenMemHandle'))
        # bind caller to cuIpcOpenMemHandle that fixes the ABI
        call_proto = ctypes.CFUNCTYPE(c_int,
                                      ctypes.POINTER(drvapi.cu_device_ptr),
                                      ctypes.POINTER(drvapi.cu_ipc_mem_handle),
                                      ctypes.c_uint)
        call_cuIpcOpenMemHandle = call_proto(_extras.call_cuIpcOpenMemHandle)
        call_cuIpcOpenMemHandle.__name__ = 'call_cuIpcOpenMemHandle'
        safe_call = self._wrap_api_call('call_cuIpcOpenMemHandle',
                                        call_cuIpcOpenMemHandle)
        # override cuIpcOpenMemHandle
        self.cuIpcOpenMemHandle = safe_call

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

        safe_call = self._wrap_api_call(fname, libfn)
        setattr(self, fname, safe_call)
        return safe_call

    def _wrap_api_call(self, fname, libfn):
        @functools.wraps(libfn)
        def safe_cuda_api_call(*args):
            _logger.debug('call driver api: %s', libfn.__name__)
            retcode = libfn(*args)
            self._check_error(fname, retcode)
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
            _logger.error(msg)
            if retcode == enums.CUDA_ERROR_NOT_INITIALIZED:
                # Detect forking
                if self.pid is not None and _getpid() != self.pid:
                    msg = 'pid %s forked from pid %s after CUDA driver init'
                    _logger.critical(msg, _getpid(), self.pid)
                    raise CudaDriverError("CUDA initialized before forking")
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

    def pop_active_context(self):
        """Pop the active CUDA context and return the handle.
        If no CUDA context is active, return None.
        """
        with self.get_active_context() as ac:
            if ac.devnum is not None:
                popped = drvapi.cu_context()
                driver.cuCtxPopCurrent(byref(popped))
                return popped

    def get_active_context(self):
        """Returns an instance of ``_ActiveContext``.
        """
        return _ActiveContext()


class _ActiveContext(object):
    """An contextmanager object to cache active context to reduce dependency
    on querying the CUDA driver API.

    Once entering the context, it is assumed that the active CUDA context is
    not changed until the context is exited.
    """
    _tls_cache = threading.local()

    def __enter__(self):
        is_top = False
        # check TLS cache
        if hasattr(self._tls_cache, 'ctx_devnum'):
            hctx, devnum = self._tls_cache.ctx_devnum
        # Not cached. Query the driver API.
        else:
            hctx = drvapi.cu_context(0)
            driver.cuCtxGetCurrent(byref(hctx))
            hctx = hctx if hctx.value else None

            if hctx is None:
                devnum = None
            else:
                hdevice = drvapi.cu_device()
                driver.cuCtxGetDevice(byref(hdevice))
                devnum = hdevice.value

                self._tls_cache.ctx_devnum = (hctx, devnum)
                is_top = True

        self._is_top = is_top
        self.context_handle = hctx
        self.devnum = devnum
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._is_top:
            delattr(self._tls_cache, 'ctx_devnum')

    def __bool__(self):
        """Returns True is there's a valid and active CUDA context.
        """
        return self.context_handle is not None

    __nonzero__ = __bool__


driver = Driver()


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
    @classmethod
    def from_identity(self, identity):
        """Create Device object from device identity created by
        ``Device.get_device_identity()``.
        """
        for devid in range(driver.get_device_count()):
            d = driver.get_device(devid)
            if d.get_device_identity() == identity:
                return d
        else:
            errmsg = (
                "No device of {} is found. "
                "Target device may not be visible in this process."
            ).format(identity)
            raise RuntimeError(errmsg)

    def __init__(self, devnum):
        got_devnum = c_int()
        driver.cuDeviceGet(byref(got_devnum), devnum)
        assert devnum == got_devnum.value, "Driver returned another device"
        self.id = got_devnum.value
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
        self.primary_context = None

    def get_device_identity(self):
        return {
            'pci_domain_id': self.PCI_DOMAIN_ID,
            'pci_bus_id': self.PCI_BUS_ID,
            'pci_device_id': self.PCI_DEVICE_ID,
        }

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

    def get_primary_context(self):
        """
        Returns the primary context for the device.
        Note: it is not pushed to the CPU thread.
        """
        if self.primary_context is not None:
            return self.primary_context

        met_requirement_for_device(self)

        # create primary context
        hctx = drvapi.cu_context()
        driver.cuDevicePrimaryCtxRetain(byref(hctx), self.id)

        ctx = Context(weakref.proxy(self), hctx)
        self.primary_context = ctx
        return ctx

    def release_primary_context(self):
        """
        Release reference to primary context
        """
        driver.cuDevicePrimaryCtxRelease(self.id)
        self.primary_context = None

    def reset(self):
        try:
            if self.primary_context is not None:
                self.primary_context.reset()
            self.release_primary_context()
        finally:
            # reset at the driver level
            driver.cuDevicePrimaryCtxReset(self.id)


def met_requirement_for_device(device):
    if device.compute_capability < MIN_REQUIRED_CC:
        raise CudaSupportError("%s has compute capability < %s" %
                               (device, MIN_REQUIRED_CC))


class _SizeNotSet(object):
    """
    Dummy object for _PendingDeallocs when *size* is not set.
    """
    def __str__(self):
        return '?'

    def __int__(self):
        return 0

_SizeNotSet = _SizeNotSet()


class _PendingDeallocs(object):
    """
    Pending deallocations of a context (or device since we are using the primary
    context).
    """
    def __init__(self, capacity):
        self._cons = deque()
        self._disable_count = 0
        self._size = 0
        self._memory_capacity = capacity

    @property
    def _max_pending_bytes(self):
        return int(self._memory_capacity * config.CUDA_DEALLOCS_RATIO)

    def add_item(self, dtor, handle, size=_SizeNotSet):
        """
        Add a pending deallocation.

        The *dtor* arg is the destructor function that takes an argument,
        *handle*.  It is used as ``dtor(handle)``.  The *size* arg is the
        byte size of the resource added.  It is an optional argument.  Some
        resources (e.g. CUModule) has an unknown memory footprint on the device.
        """
        _logger.info('add pending dealloc: %s %s bytes', dtor.__name__, size)
        self._cons.append((dtor, handle, size))
        self._size += int(size)
        if (len(self._cons) > config.CUDA_DEALLOCS_COUNT or
                self._size > self._max_pending_bytes):
            self.clear()

    def clear(self):
        """
        Flush any pending deallocations unless it is disabled.
        Do nothing if disabled.
        """
        if not self.is_disabled:
            while self._cons:
                [dtor, handle, size] = self._cons.popleft()
                _logger.info('dealloc: %s %s bytes', dtor.__name__, size)
                dtor(handle)
            self._size = 0

    @contextlib.contextmanager
    def disable(self):
        """
        Context manager to temporarily disable flushing pending deallocation.
        This can be nested.
        """
        self._disable_count += 1
        try:
            yield
        finally:
            self._disable_count -= 1
            assert self._disable_count >= 0

    @property
    def is_disabled(self):
        return self._disable_count > 0

    def __len__(self):
        """
        Returns number of pending deallocations.
        """
        return len(self._cons)


_MemoryInfo = namedtuple("_MemoryInfo", "free,total")


class Context(object):
    """
    This object wraps a CUDA Context resource.

    Contexts should not be constructed directly by user code.
    """

    def __init__(self, device, handle):
        self.device = device
        self.handle = handle
        self.allocations = utils.UniqueDict()
        # *deallocations* is lazily initialized on context push
        self.deallocations = None
        self.modules = utils.UniqueDict()
        # For storing context specific data
        self.extras = {}

    def reset(self):
        """
        Clean up all owned resources in this context.
        """
        # Free owned resources
        _logger.info('reset context of device %s', self.device.id)
        self.allocations.clear()
        self.modules.clear()
        # Clear trash
        if self.deallocations:
            self.deallocations.clear()

    def get_memory_info(self):
        """Returns (free, total) memory in bytes in the context.
        """
        free = c_size_t()
        total = c_size_t()
        driver.cuMemGetInfo(byref(free), byref(total))
        return _MemoryInfo(free=free.value, total=total.value)

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
        :param b2d_func: function that calculates how much per-block dynamic
                         shared memory 'func' uses based on the block size.
                         Can also be the address of a C function.
                         Use `0` to pass `NULL` to the underlying CUDA API.
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

    def prepare_for_use(self):
        """Initialize the context for use.
        It's safe to be called multiple times.
        """
        # setup *deallocations* as the context becomes active for the first time
        if self.deallocations is None:
            self.deallocations = _PendingDeallocs(self.get_memory_info().total)

    def push(self):
        """
        Pushes this context on the current CPU Thread.
        """
        driver.cuCtxPushCurrent(self.handle)
        self.prepare_for_use()

    def pop(self):
        """
        Pops this context off the current CPU thread. Note that this context must
        be at the top of the context stack, otherwise an error will occur.
        """
        popped = driver.pop_active_context()
        assert popped.value == self.handle.value

    def _attempt_allocation(self, allocator):
        """
        Attempt allocation by calling *allocator*.  If a out-of-memory error
        is raised, the pending deallocations are flushed and the allocation
        is retried.  If it fails in the second attempt, the error is reraised.
        """
        try:
            allocator()
        except CudaAPIError as e:
            # is out-of-memory?
            if e.code == enums.CUDA_ERROR_OUT_OF_MEMORY:
                # clear pending deallocations
                self.deallocations.clear()
                # try again
                allocator()
            else:
                raise

    def memalloc(self, bytesize):
        ptr = drvapi.cu_device_ptr()

        def allocator():
            driver.cuMemAlloc(byref(ptr), bytesize)

        self._attempt_allocation(allocator)

        finalizer = _alloc_finalizer(self, ptr, bytesize)
        mem = AutoFreePointer(weakref.proxy(self), ptr, bytesize, finalizer)
        self.allocations[ptr.value] = mem
        return mem.own()

    def memhostalloc(self, bytesize, mapped=False, portable=False, wc=False):
        pointer = c_void_p()
        flags = 0
        if mapped:
            flags |= enums.CU_MEMHOSTALLOC_DEVICEMAP
        if portable:
            flags |= enums.CU_MEMHOSTALLOC_PORTABLE
        if wc:
            flags |= enums.CU_MEMHOSTALLOC_WRITECOMBINED

        def allocator():
            driver.cuMemHostAlloc(byref(pointer), bytesize, flags)

        if mapped:
            self._attempt_allocation(allocator)
        else:
            allocator()

        owner = None

        finalizer = _hostalloc_finalizer(self, pointer, bytesize, mapped)

        if mapped:
            mem = MappedMemory(weakref.proxy(self), owner, pointer, bytesize,
                               finalizer=finalizer)
            self.allocations[mem.handle.value] = mem
            return mem.own()
        else:
            mem = PinnedMemory(weakref.proxy(self), owner, pointer, bytesize,
                               finalizer=finalizer)
            return mem

    def mempin(self, owner, pointer, size, mapped=False):
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

        def allocator():
            driver.cuMemHostRegister(pointer, size, flags)

        if mapped:
            self._attempt_allocation(allocator)
        else:
            allocator()

        finalizer = _pin_finalizer(self, pointer, mapped)

        if mapped:
            mem = MappedMemory(weakref.proxy(self), owner, pointer, size,
                               finalizer=finalizer)
            self.allocations[mem.handle.value] = mem
            return mem.own()
        else:
            mem = PinnedMemory(weakref.proxy(self), owner, pointer, size,
                               finalizer=finalizer)
            return mem

    def memunpin(self, pointer):
        raise NotImplementedError

    def get_ipc_handle(self, memory):
        """
        Returns a *IpcHandle* from a GPU allocation.
        """
        if not SUPPORTS_IPC:
            raise OSError('OS does not support CUDA IPC')
        ipchandle = drvapi.cu_ipc_mem_handle()
        driver.cuIpcGetMemHandle(
            ctypes.byref(ipchandle),
            memory.owner.handle,
            )
        source_info = self.device.get_device_identity()
        offset = memory.handle.value - memory.owner.handle.value
        return IpcHandle(memory, ipchandle, memory.size, source_info,
                         offset=offset)

    def open_ipc_handle(self, handle, size):
        # open the IPC handle to get the device pointer
        dptr = drvapi.cu_device_ptr()
        flags = 1  # CU_IPC_MEM_LAZY_ENABLE_PEER_ACCESS
        driver.cuIpcOpenMemHandle(byref(dptr), handle, flags)
        # wrap it
        return MemoryPointer(context=weakref.proxy(self), pointer=dptr,
                             size=size)

    def enable_peer_access(self, peer_context, flags=0):
        """Enable peer access between the current context and the peer context
        """
        assert flags == 0, '*flags* is reserved and MUST be zero'
        driver.cuCtxEnablePeerAccess(peer_context, flags)

    def can_access_peer(self, peer_device):
        """Returns a bool indicating whether the peer access between the
        current and peer device is possible.
        """
        can_access_peer = c_int()
        driver.cuDeviceCanAccessPeer(
            byref(can_access_peer),
            self.device.id,
            peer_device,
            )
        return bool(can_access_peer)

    def create_module_ptx(self, ptx):
        if isinstance(ptx, str):
            ptx = ptx.encode('utf8')
        image = c_char_p(ptx)
        return self.create_module_image(image)

    def create_module_image(self, image):
        module = load_module_image(self, image)
        self.modules[module.handle.value] = module
        return weakref.proxy(module)

    def unload_module(self, module):
        del self.modules[module.handle.value]

    def create_stream(self):
        handle = drvapi.cu_stream()
        driver.cuStreamCreate(byref(handle), 0)
        return Stream(weakref.proxy(self), handle,
                      _stream_finalizer(self.deallocations, handle))

    def create_event(self, timing=True):
        handle = drvapi.cu_event()
        flags = 0
        if not timing:
            flags |= enums.CU_EVENT_DISABLE_TIMING
        driver.cuEventCreate(byref(handle), flags)
        return Event(weakref.proxy(self), handle,
                     finalizer=_event_finalizer(self.deallocations, handle))

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
    logsz = int(get_numba_envvar('CUDA_LOG_SIZE', 1024))

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


def _alloc_finalizer(context, handle, size):
    allocations = context.allocations
    deallocations = context.deallocations

    def core():
        if allocations:
            del allocations[handle.value]
        deallocations.add_item(driver.cuMemFree, handle, size)

    return core


def _hostalloc_finalizer(context, handle, size, mapped):
    """
    Finalize page-locked host memory allocated by `context.memhostalloc`.

    This memory is managed by CUDA, and finalization entails deallocation. The
    issues noted in `_pin_finalizer` are not relevant in this case, and the
    finalization is placed in the `context.deallocations` queue along with
    finalization of device objects.

    """
    allocations = context.allocations
    deallocations = context.deallocations
    if not mapped:
        size = _SizeNotSet

    def core():
        if mapped and allocations:
            del allocations[handle.value]
        deallocations.add_item(driver.cuMemFreeHost, handle, size)

    return core


def _pin_finalizer(context, handle, mapped):
    """
    Finalize temporary page-locking of host memory by `context.mempin`.

    This applies to memory not otherwise managed by CUDA. Page-locking can
    be requested multiple times on the same memory, and must therefore be
    lifted as soon as finalization is requested, otherwise subsequent calls to
    `mempin` may fail with `CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED`, leading
    to unexpected behavior for the context managers `cuda.{pinned,mapped}`.
    This function therefore carries out finalization immediately, bypassing the
    `context.deallocations` queue.

    """
    allocations = context.allocations

    def core():
        if mapped and allocations:
            del allocations[handle.value]
        driver.cuMemHostUnregister(handle)

    return core


def _event_finalizer(deallocs, handle):
    def core():
        deallocs.add_item(driver.cuEventDestroy, handle)

    return core


def _stream_finalizer(deallocs, handle):
    def core():
        deallocs.add_item(driver.cuStreamDestroy, handle)

    return core


def _module_finalizer(context, handle):
    dealloc = context.deallocations
    modules = context.modules

    def core():
        shutting_down = utils.shutting_down  # early bind

        def module_unload(handle):
            # If we are not shutting down, we must be called due to
            # Context.reset() of Context.unload_module().  Both must have
            # cleared the module reference from the context.
            assert shutting_down() or handle.value not in modules
            driver.cuModuleUnload(handle)

        dealloc.add_item(module_unload, handle)

    return core


class _CudaIpcImpl(object):
    """Implementation of GPU IPC using CUDA driver API.
    This requires the devices to be peer accessible.
    """
    def __init__(self, parent):
        self.base = parent.base
        self.handle = parent.handle
        self.size = parent.size
        self.offset = parent.offset
        # remember if the handle is already opened
        self._opened_mem = None

    def open(self, context):
        """
        Import the IPC memory and returns a raw CUDA memory pointer object
        """
        if self.base is not None:
            raise ValueError('opening IpcHandle from original process')

        if self._opened_mem is not None:
            raise ValueError('IpcHandle is already opened')

        mem = context.open_ipc_handle(self.handle, self.offset + self.size)
        # this object owns the opened allocation
        # note: it is required the memory be freed after the ipc handle is
        #       closed by the importing context.
        self._opened_mem = mem
        return mem.own().view(self.offset)

    def close(self):
        if self._opened_mem is None:
            raise ValueError('IpcHandle not opened')
        driver.cuIpcCloseMemHandle(self._opened_mem.handle)
        self._opened_mem = None


class _StagedIpcImpl(object):
    """Implementation of GPU IPC using custom staging logic to workaround
    CUDA IPC limitation on peer accessibility between devices.
    """
    def __init__(self, parent, source_info):
        self.parent = parent
        self.base = parent.base
        self.handle = parent.handle
        self.size = parent.size
        self.source_info = source_info

    def open(self, context):
        from numba import cuda

        srcdev = Device.from_identity(self.source_info)

        impl = _CudaIpcImpl(parent=self.parent)
        # Open context on the source device.
        with cuda.gpus[srcdev.id]:
            source_ptr = impl.open(cuda.devices.get_context())

        # Allocate GPU buffer.
        newmem = context.memalloc(self.size)
        # Do D->D from the source peer-context
        # This performs automatic host staging
        device_to_device(newmem, source_ptr, self.size)

        # Cleanup source context
        with cuda.gpus[srcdev.id]:
            impl.close()

        return newmem.own()

    def close(self):
        # Nothing has to be done here
        pass


class IpcHandle(object):
    """
    Internal IPC handle.

    Serialization of the CUDA IPC handle object is implemented here.

    The *base* attribute is a reference to the original allocation to keep it
    alive.  The *handle* is a ctypes object of the CUDA IPC handle. The *size*
    is the allocation size.
    """
    def __init__(self, base, handle, size, source_info=None, offset=0):
        self.base = base
        self.handle = handle
        self.size = size
        self.source_info = source_info
        self._impl = None
        self.offset = offset

    def _sentry_source_info(self):
        if self.source_info is None:
            raise RuntimeError("IPC handle doesn't have source info")

    def can_access_peer(self, context):
        """Returns a bool indicating whether the active context can peer
        access the IPC handle
        """
        self._sentry_source_info()
        if self.source_info == context.device.get_device_identity():
            return True
        source_device = Device.from_identity(self.source_info)
        return context.can_access_peer(source_device.id)

    def open_staged(self, context):
        """Open the IPC by allowing staging on the host memory first.
        """
        self._sentry_source_info()

        if self._impl is not None:
            raise ValueError('IpcHandle is already opened')

        self._impl = _StagedIpcImpl(self, self.source_info)
        return self._impl.open(context)

    def open_direct(self, context):
        """
        Import the IPC memory and returns a raw CUDA memory pointer object
        """
        if self._impl is not None:
            raise ValueError('IpcHandle is already opened')

        self._impl = _CudaIpcImpl(self)
        return self._impl.open(context)

    def open(self, context):
        """Open the IPC handle and import the memory for usage in the given
        context.  Returns a raw CUDA memory pointer object.

        This is enhanced over CUDA IPC that it will work regardless of whether
        the source device is peer-accessible by the destination device.
        If the devices are peer-accessible, it uses .open_direct().
        If the devices are not peer-accessible, it uses .open_staged().
        """
        if self.source_info is None or self.can_access_peer(context):
            fn = self.open_direct
        else:
            fn = self.open_staged
        return fn(context)

    def open_array(self, context, shape, dtype, strides=None):
        """
        Similar to `.open()` but returns an device array.
        """
        from . import devicearray

        # by default, set strides to itemsize
        if strides is None:
            strides = dtype.itemsize
        dptr = self.open(context)
        # read the device pointer as an array
        return devicearray.DeviceNDArray(shape=shape, strides=strides,
                                         dtype=dtype, gpu_data=dptr)

    def close(self):
        if self._impl is None:
            raise ValueError('IpcHandle not opened')
        self._impl.close()
        self._impl = None

    def __reduce__(self):
        # Preprocess the IPC handle, which is defined as a byte array.
        preprocessed_handle = tuple(self.handle)
        args = (
            self.__class__,
            preprocessed_handle,
            self.size,
            self.source_info,
            self.offset,
            )
        return (serialize._rebuild_reduction, args)

    @classmethod
    def _rebuild(cls, handle_ary, size, source_info, offset):
        handle = drvapi.cu_ipc_mem_handle(*handle_ary)
        return cls(base=None, handle=handle, size=size,
                   source_info=source_info, offset=offset)


class MemoryPointer(object):
    """A memory pointer that owns the buffer with an optional finalizer.

    When an instance is deleted, the finalizer will be called regardless
    of the `.refct`.

    An instance is created with `.refct=1`.  The buffer lifetime
    is tied to the MemoryPointer instance's lifetime.  The finalizer is invoked
    only if the MemoryPointer instance's lifetime ends.
    """
    __cuda_memory__ = True

    def __init__(self, context, pointer, size, finalizer=None, owner=None):
        self.context = context
        self.device_pointer = pointer
        self.size = size
        self._cuda_memsize_ = size
        self.is_managed = finalizer is not None
        self.refct = 1
        self.handle = self.device_pointer
        self._owner = owner

        if finalizer is not None:
            self._finalizer = weakref.finalize(self, finalizer)

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
        if stop is None:
            size = self.size - start
        else:
            size = stop - start

        # Handle NULL/empty memory buffer
        if self.device_pointer.value is None:
            if size != 0:
                raise RuntimeError("non-empty slice into empty slice")
            view = self      # new view is just a reference to self
        # Handle normal case
        else:
            base = self.device_pointer.value + start
            if size < 0:
                raise RuntimeError('size cannot be negative')
            pointer = drvapi.cu_device_ptr(base)
            view = MemoryPointer(self.context, pointer, size, owner=self.owner)

        if isinstance(self.owner, (MemoryPointer, OwnedPointer)):
            # Owned by a numba-managed memory segment, take an owned reference
            return OwnedPointer(weakref.proxy(self.owner), view)
        else:
            # Owned by external alloc, return view with same external owner
            return view

    @property
    def device_ctypes_pointer(self):
        return self.device_pointer


class AutoFreePointer(MemoryPointer):
    """Modifies the ownership semantic of the MemoryPointer so that the
    instance lifetime is directly tied to the number of references.

    When `.refct` reaches zero, the finalizer is invoked.
    """
    def __init__(self, *args, **kwargs):
        super(AutoFreePointer, self).__init__(*args, **kwargs)
        # Releease the self reference to the buffer, so that the finalizer
        # is invoked if all the derived pointers are gone.
        self.refct -= 1


class MappedMemory(AutoFreePointer):
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
            weakref.finalize(self, finalizer)

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
            try:
                mem.refct -= 1
                assert mem.refct >= 0
                if mem.refct == 0:
                    mem.free()
            except ReferenceError:
                # ignore reference error here
                pass

        self._mem.refct += 1
        weakref.finalize(self, deref)

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
            weakref.finalize(self, finalizer)

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
            weakref.finalize(self, finalizer)

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
            self._finalizer = weakref.finalize(self, finalizer)

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
    def __init__(self, max_registers=0):
        logsz = int(get_numba_envvar('CUDA_LOG_SIZE', 1024))
        linkerinfo = (c_char * logsz)()
        linkererrors = (c_char * logsz)()

        options = {
            enums.CU_JIT_INFO_LOG_BUFFER: addressof(linkerinfo),
            enums.CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES: c_void_p(logsz),
            enums.CU_JIT_ERROR_LOG_BUFFER: addressof(linkererrors),
            enums.CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES: c_void_p(logsz),
            enums.CU_JIT_LOG_VERBOSE: c_void_p(1),
        }
        if max_registers:
            options[enums.CU_JIT_MAX_REGISTERS] = c_void_p(max_registers)

        raw_keys = list(options.keys()) + [enums.CU_JIT_TARGET_FROM_CUCONTEXT]
        raw_values = list(options.values())
        del options

        option_keys = (drvapi.cu_jit_option * len(raw_keys))(*raw_keys)
        option_vals = (c_void_p * len(raw_values))(*raw_values)

        self.handle = handle = drvapi.cu_link_state()
        driver.cuLinkCreate(len(raw_keys), option_keys, option_vals,
                            byref(self.handle))

        weakref.finalize(self, driver.cuLinkDestroy, handle)

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


def get_devptr_for_active_ctx(ptr):
    """Query the device pointer usable in the current context from an arbitrary
    pointer.
    """
    devptr = c_void_p(0)
    if ptr != 0:
        attr = enums.CU_POINTER_ATTRIBUTE_DEVICE_POINTER
        driver.cuPointerGetAttribute(byref(devptr), attr, ptr)
    return devptr


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
    assert sz >= 0, "{} length array".format(sz)
    return sz


def _is_datetime_dtype(obj):
    """Returns True if the obj.dtype is datetime64 or timedelta64
    """
    dtype = getattr(obj, 'dtype', None)
    return dtype is not None and dtype.char in 'Mm'


def _workaround_for_datetime(obj):
    """Workaround for numpy#4983: buffer protocol doesn't support
    datetime64 or timedelta64.
    """
    if _is_datetime_dtype(obj):
        obj = obj.view(np.int64)
    return obj

def host_pointer(obj, readonly=False):
    """Get host pointer from an obj.

    If `readonly` is False, the buffer must be writable.

    NOTE: The underlying data pointer from the host data buffer is used and
    it should not be changed until the operation which can be asynchronous
    completes.
    """
    if isinstance(obj, (int, long)):
        return obj

    forcewritable = False
    if not readonly:
        forcewritable = isinstance(obj, np.void) or _is_datetime_dtype(obj)

    obj = _workaround_for_datetime(obj)
    return mviewbuf.memoryview_get_buffer(obj, forcewritable, readonly)

def host_memory_extents(obj):
    "Returns (start, end) the start and end pointer of the array (half open)."
    obj = _workaround_for_datetime(obj)
    return mviewbuf.memoryview_get_extents(obj)


def memory_size_from_info(shape, strides, itemsize):
    """Get the byte size of a contiguous memory buffer given the shape, strides
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

    fn(device_pointer(dst), host_pointer(src, readonly=True), size, *varargs)


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
