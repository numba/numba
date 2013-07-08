import contextlib
import numpy as np

from .cudadrv import initialize as _initialize
from .cudadrv.driver import require_context
from .cudadrv import devicearray, driver

from .cudapy.ptx import (threadIdx, blockIdx, blockDim, gridDim, syncthreads,
                         shared, grid, atomic)
from .cudapy import jit, autojit, declare_device

# NDarray device helper
@require_context
def to_device(ary, stream=0, copy=True, to=None):
    """Allocate and transfer a numpy ndarray to the device.
    
    To copy host->device a numpy array::
        
        ary = numpy.arange(10)
        d_ary = cuda.to_device(ary)
        
    To enqueue the transfer to a stream
        
        stream = cuda.stream()
        d_ary = cuda.to_device(ary, stream=stream)
        
    The resulting ``d_ary`` is a ``DeviceNDArray``.
    
    To copy device->host::
    
        hary = d_ary.copy_to_host()

    To copy device->host to an existing array::
    
        ary = numpy.empty(shape=d_ary.shape, dtype=d_ary.dtype)
        d_ary.copy_to_host(ary)

    To enqueue the transfer to a stream::
    
        hary = d_ary.copy_to_host(stream=stream)
    """
    if to is None:
        devarray = devicearray.from_array_like(ary, stream=stream)
    else:
        devarray = to
    if copy:
        devarray.copy_to_device(ary, stream=stream)
    return devarray

@require_context
def device_array(shape, dtype=np.float, strides=None, order='C', stream=0):
    """Allocate a device ndarray like numpy.empty()
    """
    shape, strides, dtype = _prepare_shape_strides_dtype(shape, strides, dtype,
                                                         order)
    return devicearray.DeviceNDArray(shape=shape, strides=strides, dtype=dtype,
                                     stream=stream)

@require_context
def pinned_array(shape, dtype=np.float, strides=None, order='C'):
    """Allocate a numpy.ndarray with a buffer that is pinned (pagelocked).
    Similar to numpy.empty().
    """
    shape, strides, dtype = _prepare_shape_strides_dtype(shape, strides, dtype,
                                                         order)
    bytesize = driver.memory_size_from_info(shape, strides, dtype.itemsize)
    buffer = driver.HostAllocMemory(bytesize)
    return np.ndarray(shape=shape, strides=strides, dtype=dtype, order=order,
                      buffer=buffer)

@require_context
def mapped_array(shape, dtype=np.float, strides=None, order='C', stream=0,
                 portable=False, wc=False):
    """Allocate a mapped ndarray with a buffer that is pinned and mapped on
    to the device. Similar to numpy.empty()
    
    portable: a boolean flag to allow the allocated device memory to be 
              usable in multiple devices.
    wc: a boolean flag to enable writecombined allocation which is faster
        to write by the host and to read by the device, but slower to 
        write by the host and slower to write by the device.
    """
    shape, strides, dtype = _prepare_shape_strides_dtype(shape, strides, dtype,
                                                         order)
    bytesize = driver.memory_size_from_info(shape, strides, dtype.itemsize)
    buffer = driver.HostAllocMemory(bytesize, mapped=True)
    npary = np.ndarray(shape=shape, strides=strides, dtype=dtype, order=order,
                       buffer=buffer)
    mappedview = np.ndarray.view(npary, type=devicearray.MappedNDArray)
    mappedview.device_setup(buffer, stream=stream)
    return mappedview

def synchronize():
    "Synchronize current context"
    drv = driver.Driver()
    return drv.current_context().synchronize()

def _prepare_shape_strides_dtype(shape, strides, dtype, order):
    dtype = np.dtype(dtype)
    if isinstance(shape, (int, long)):
        shape = (shape,)
    if isinstance(strides, (int, long)):
        strides = (strides,)
    else:
        strides = strides or _fill_stride_by_order(shape, dtype, order)
    return shape, strides, dtype

def _fill_stride_by_order(shape, dtype, order):
    nd = len(shape)
    strides = [0] * nd
    if order == 'C':
        strides[-1] = dtype.itemsize
        for d in reversed(range(nd - 1)):
            strides[d] = strides[d + 1] * shape[d + 1]
    elif order == 'F':
        strides[0] = dtype.itemsize
        for d in range(1, nd):
            strides[d] = strides[d - 1] * shape[d - 1]
    else:
        raise ValueError('must be either C/F order')
    return tuple(strides)


def device_array_like(ary, stream=0):
    """Call cuda.devicearray() with information from the array.
    """
    return device_array(shape=ary.shape, dtype=ary.dtype,
                        strides=ary.strides, stream=stream)

# Stream helper
@require_context
def stream():
    """Create a CUDA stream that represents a command queue for the device.
    """
    return driver.Stream()

# Page lock
@require_context
@contextlib.contextmanager
def pinned(*arylist):
    """A context manager for temporary pinning a sequence of host ndarrays.
    """
    pmlist = []
    for ary in arylist:
        pm = driver.PinnedMemory(ary, driver.host_pointer(ary),
                                 driver.host_memory_size(ary), mapped=False)
        pmlist.append(pm)
    yield
    del pmlist


@require_context
@contextlib.contextmanager
def mapped(*arylist, **kws):
    """A context manager for temporarily mapping a sequence of host ndarrays.
    """
    assert not kws or 'stream' in kws, "Only accept 'stream' as keyword."
    pmlist = []
    stream = kws.get('stream', 0)
    for ary in arylist:
        pm = driver.PinnedMemory(ary, driver.host_pointer(ary),
                                 driver.host_memory_size(ary), mapped=True)
        pmlist.append(pm)

    devarylist = []
    for ary, pm in zip(arylist, pmlist):
        devary = devicearray.from_array_like(ary, gpu_data=pm, stream=stream)
        devarylist.append(devary)
    if len(devarylist) == 1:
        yield devarylist[0]
    else:
        yield devarylist


def event(timing=True):
    """Create a CUDA event.
    """
    evt = driver.Event(timing=timing)
    return evt

# Device selection

def select_device(device_id):
    '''Creates a new CUDA context with the selected device.
    The context is associated with the current thread.
    NumbaPro currently allows only one context per thread.

    Returns a device instance

    Raises exception on error.
    '''
    drv = driver.Driver()
    device = driver.Device(device_id)
    drv.create_context(device)
    return device

def get_current_device():
    "Get current device associated with the current thread"
    drv = driver.Driver()
    return drv.current_context().device

def list_devices():
    "List all CUDA devices"
    drv = driver.Driver()
    return [driver.Device(i) for i in range(drv.get_device_count())]

def close():
    """Explicitly closes the context.

    Destroy the current context of the current thread
    """
    drv = driver.Driver()
    drv.release_context(drv.current_context())

def _auto_device(ary, stream=0, copy=True):
    return devicearray.auto_device(ary, stream=stream, copy=copy)

#
# Initialize the CUDA system
#
is_available = _initialize.initialize()

