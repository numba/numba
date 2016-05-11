"""
API that are reported to numba.cuda
"""

from __future__ import print_function, absolute_import

import contextlib

import numpy as np

from .cudadrv import devicearray, devices, driver


try:
    long
except NameError:
    long = int

# NDarray device helper

require_context = devices.require_context
current_context = devices.get_context
gpus = devices.gpus


@require_context
def to_device(obj, stream=0, copy=True, to=None):
    """to_device(obj, stream=0, copy=True, to=None)

    Allocate and transfer a numpy ndarray or structured scalar to the device.

    To copy host->device a numpy array::

        ary = np.arange(10)
        d_ary = cuda.to_device(ary)

    To enqueue the transfer to a stream::

        stream = cuda.stream()
        d_ary = cuda.to_device(ary, stream=stream)

    The resulting ``d_ary`` is a ``DeviceNDArray``.

    To copy device->host::

        hary = d_ary.copy_to_host()

    To copy device->host to an existing array::

        ary = np.empty(shape=d_ary.shape, dtype=d_ary.dtype)
        d_ary.copy_to_host(ary)

    To enqueue the transfer to a stream::

        hary = d_ary.copy_to_host(stream=stream)
    """
    if to is None:
        to, new = devicearray.auto_device(obj, stream=stream, copy=copy)
        return to
    if copy:
        to.copy_to_device(obj, stream=stream)
    return to


@require_context
def device_array(shape, dtype=np.float, strides=None, order='C', stream=0):
    """device_array(shape, dtype=np.float, strides=None, order='C', stream=0)

    Allocate an empty device ndarray. Similar to :meth:`numpy.empty`.
    """
    shape, strides, dtype = _prepare_shape_strides_dtype(shape, strides, dtype,
                                                         order)
    return devicearray.DeviceNDArray(shape=shape, strides=strides, dtype=dtype,
                                     stream=stream)


@require_context
def pinned_array(shape, dtype=np.float, strides=None, order='C'):
    """pinned_array(shape, dtype=np.float, strides=None, order='C')

    Allocate a np.ndarray with a buffer that is pinned (pagelocked).
    Similar to np.empty().
    """
    shape, strides, dtype = _prepare_shape_strides_dtype(shape, strides, dtype,
                                                         order)
    bytesize = driver.memory_size_from_info(shape, strides,
                                            dtype.itemsize)
    buffer = current_context().memhostalloc(bytesize)
    return np.ndarray(shape=shape, strides=strides, dtype=dtype, order=order,
                      buffer=buffer)


@require_context
def mapped_array(shape, dtype=np.float, strides=None, order='C', stream=0,
                 portable=False, wc=False):
    """mapped_array(shape, dtype=np.float, strides=None, order='C', stream=0, portable=False, wc=False)

    Allocate a mapped ndarray with a buffer that is pinned and mapped on
    to the device. Similar to np.empty()

    :param portable: a boolean flag to allow the allocated device memory to be
              usable in multiple devices.
    :param wc: a boolean flag to enable writecombined allocation which is faster
        to write by the host and to read by the device, but slower to
        write by the host and slower to write by the device.
    """
    shape, strides, dtype = _prepare_shape_strides_dtype(shape, strides, dtype,
                                                         order)
    bytesize = driver.memory_size_from_info(shape, strides, dtype.itemsize)
    buffer = current_context().memhostalloc(bytesize, mapped=True)
    npary = np.ndarray(shape=shape, strides=strides, dtype=dtype, order=order,
                       buffer=buffer)
    mappedview = np.ndarray.view(npary, type=devicearray.MappedNDArray)
    mappedview.device_setup(buffer, stream=stream)
    return mappedview


def synchronize():
    "Synchronize the current context."
    return current_context().synchronize()


def _prepare_shape_strides_dtype(shape, strides, dtype, order):
    dtype = np.dtype(dtype)
    if isinstance(shape, (int, long)):
        shape = (shape,)
    if isinstance(strides, (int, long)):
        strides = (strides,)
    else:
        if shape == ():
            shape = (1,)
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
    """stream()

    Create a CUDA stream that represents a command queue for the device.
    """
    return current_context().create_stream()

# Page lock
@require_context
@contextlib.contextmanager
def pinned(*arylist):
    """A context manager for temporary pinning a sequence of host ndarrays.
    """
    pmlist = []
    for ary in arylist:
        pm = current_context().mempin(ary, driver.host_pointer(ary),
                                      driver.host_memory_size(ary),
                                      mapped=False)
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
        pm = current_context().mempin(ary, driver.host_pointer(ary),
                                    driver.host_memory_size(ary),
                                    mapped=True)
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
    """
    Create a CUDA event. Timing data is only recorded by the event if it is
    created with ``timing=True``.
    """
    evt = current_context().create_event(timing=timing)
    return evt

event_elapsed_time = driver.event_elapsed_time

# Device selection

def select_device(device_id):
    """
    Make the context associated with device *device_id* the current context.

    Returns a Device instance.

    Raises exception on error.
    """
    context = devices.get_context(device_id)
    return context.device


def get_current_device():
    "Get current device associated with the current thread"
    return current_context().device


def list_devices():
    "Return a list of all detected devices"
    return devices.gpus


def close():
    """
    Explicitly clears all contexts in the current thread, and destroys all
    contexts if the current thread is the main thread.
    """
    devices.reset()


def _auto_device(ary, stream=0, copy=True):
    return devicearray.auto_device(ary, stream=stream, copy=copy)


def detect():
    """
    Detect supported CUDA hardware and print a summary of the detected hardware.

    Returns a boolean indicating whether any supported devices were detected.
    """
    devlist = list_devices()
    print('Found %d CUDA devices' % len(devlist))
    supported_count = 0
    for dev in devlist:
        attrs = []
        cc = dev.compute_capability
        attrs += [('compute capability', '%d.%d' % cc)]
        attrs += [('pci device id', dev.PCI_DEVICE_ID)]
        attrs += [('pci bus id', dev.PCI_BUS_ID)]
        if cc < (2, 0):
            support = '[NOT SUPPORTED: CC < 2.0]'
        else:
            support = '[SUPPORTED]'
            supported_count += 1

        print('id %d    %20s %40s' % (dev.id, dev.name, support))
        for key, val in attrs:
            print('%40s: %s' % (key, val))

    print('Summary:')
    print('\t%d/%d devices are supported' % (supported_count, len(devlist)))
    return supported_count > 0


@contextlib.contextmanager
def defer_cleanup():
    tserv = get_current_device().trashing
    with tserv.defer_cleanup:
        yield


profiling = require_context(driver.profiling)
profile_start = require_context(driver.profile_start)
profile_stop = require_context(driver.profile_stop)
