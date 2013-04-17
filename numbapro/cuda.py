import contextlib
import numpy as np

from .cudapipeline import initialize as _initialize
from .cudapipeline.special_values import *
from .cudapipeline.driver import require_context
from .cudapipeline import devicearray
from .cudapipeline.decorators import jit, autojit

# NDarray device helper
@require_context
def to_device(ary, stream=0, copy=True, to=None):
    if to is None:
        devarray = devicearray.from_array_like(ary, stream=stream)
    else:
        devarray = to
    if copy:
        devarray.copy_to_device(ary, stream=stream)
    return devarray

@require_context
def device_array(shape, dtype=np.float, strides=None, order='C', stream=0):
    dtype = np.dtype(dtype)
    if isinstance(shape, (int, long)):
        shape = (shape,)
    if not strides:
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
    return devicearray.DeviceNDArray(shape, strides, dtype, stream=stream)

def device_array_like(ary, stream=0):
    order = ''
    if ary.flags['C_CONTIGUOUS']:
        order = 'C'
    elif ary.flags['F_CONTIGUOUS']:
        order = 'F'
    return device_array(shape=ary.shape, dtype=ary.dtype,
                        strides=ary.strides, stream=stream)

# Stream helper
@require_context
def stream():
    from numbapro.cudapipeline.driver import Stream
    return Stream()

# Page lock
@require_context
@contextlib.contextmanager
def pinned(*arylist):
    from numbapro.cudapipeline.driver import PinnedMemory, host_memory_size, host_pointer
    pmlist = []
    for ary in arylist:
        pm = PinnedMemory(ary, host_pointer(ary), host_memory_size(ary),
                          mapped=False)
        pmlist.append(pm)
    yield
    del pmlist


@require_context
@contextlib.contextmanager
def mapped(*arylist, **kws):
    assert not kws or 'stream' in kws, "Only accept 'stream' as keyword."
    from numbapro.cudapipeline.driver import PinnedMemory, host_memory_size, host_pointer, device_pointer
    pmlist = []
    stream = kws.get('stream', 0)
    for ary in arylist:
        pm = PinnedMemory(ary, host_pointer(ary), host_memory_size(ary),
                          mapped=True)
        pmlist.append(pm)

    devarylist = []
    for ary, pm in zip(arylist, pmlist):
        dptr = device_pointer(pm)
        devary = devicearray.from_array_like(ary, gpu_data=pm, stream=stream)
        devarylist.append(devary)
    if len(devarylist) == 1:
        yield devarylist[0]
    else:
        yield devarylist


def event(timing=True):
    from numbapro.cudapipeline.driver import Event
    evt = Event(timing=timing)
    return evt

# Device selection

def select_device(device_id):
    '''Call this before any CUDA feature is used in each thread.

        Returns a device instance

        Raises exception on error.
        '''
    from numbapro.cudapipeline import driver as cu
    driver = cu.Driver()
    device = cu.Device(device_id)
    context = driver.create_context(device)
    return device

def get_current_device():
    "Get current device associated with the current thread"
    from numbapro.cudapipeline import driver
    driver = driver.Driver()
    return driver.current_context().device

def close():
    '''Explicitly closes the context.

        Destroy the current context of the current thread
        '''
    from numbapro.cudapipeline import driver as cu
    driver = cu.Driver()
    driver.release_context(driver.current_context())

def _auto_device(ary, stream=0, copy=True):
    if devicearray.is_cuda_ndarray(ary):
        return ary, False
    else:
        return to_device(ary, copy=copy, stream=stream), True


#
# Initialize the CUDA system
#
is_available = _initialize.initialize()

