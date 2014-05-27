"""
API for OpenCL

This aims to be compatible with the cuda one. It will reuse
the naming convention used in the CUDA api, even if it mismatches
the OpenCL one.

For example:
 - stream keyword for CommandQueues
"""

from __future__ import print_function, absolute_import, division
from . import ocldrv
from .ocldrv import oclarray, oclmem, devices
from .. import mviewbuf
import numpy as np
import contextlib

try:
    long
except NameError:
    long = int

# Array to device migration ############################################

def to_device(ary, stream=None, copy=True, to=None):
    """
    to_device(ary, queue=None, copy=True, to=None)

    Allocate and transfer a NumPy ndarray in an OpenCL context.

    Parameters
    ----------
    ary : array_like
        The shape and datatype of 'ary' define the resulting device
        array. 'ary' may be the source of actual data if 'copy' is
        True
    stream: CommandQueue
        The CommandQueue in which the transfer is to be performed. This
        implies async operation
    copy : boolean
        whether the contents of 'ary' should be copied to the resulting
        array.
    to : device_array
        Instead of allocating a new device array, use the device_array
        as destination. Only useful when copy=True. In this case,
        context_or_queue MUST be a queue
    block : boolean
        if True the function won't return until the operation has
        finished. If False the function will return as soon as possible
        returning an event that can be used to check the status of
        the operation.

    Returns
    -------
    out : device_array
        The new array (or the array used as 'to' argument if one was
        provided.

    See Also
    --------

    Notes
    -----

    Examples
    --------
    """
    assert stream is None or isinstance(stream, ocldrv.CommandQueue)

    if stream:
        ctxt = stream.context
    else:
        ctxt = current_context()

    if to is None:
        if copy:
            devarray = oclarray.from_array(ary, ctxt)
        else:
            devarray = oclarray.from_array_like(ary, ctxt)
    else:
        # to is specified, just copy the array
        assert copy
        oclarray.require_ocl_ndarray(to)
        devarray = to
        if copy:
            devarray.copy_to_device(ary, queue=current_queue())

    return devarray


def device_array(context_or_queue, shape, dtype=np.float, strides=None, order='C'):
    """device_array(shape, dtype=np.float, strides=None, order='C')

    Allocate and empty device array. Similar to numpy.empty()
    """
    assert((strides is None) or (len(strides) == len(shape)))
    ctxt = context_or_queue if isinstance(context_or_queue, ocldrv.Context) else context_or_queue.context
    shape, strides, dtype = _prepare_shape_strides_dtype(shape, strides, dtype, order)
    s, e = mviewbuf.memoryview_get_extents_info(shape, strides, len(shape), dtype.itemsize)
    cl_desc = oclarray._create_ocl_desc(ctxt, shape, strides)
    cl_data = ctxt.create_buffer(e-s)
    return oclarray.OpenCLNDArray(shape, strides, dtype, cl_desc, cl_data)


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


# device memory utils
def is_device_memory(obj):
    return getattr(obj, '__ocl_memory__', False)


# Device selection

def select_device(device_id):
    """Creates a new OpenCL context with the selected device.
    The context is associated with the current thread.
    Numba currently allows only one context per thread.

    return a device instance.

    Raises exception on error.
    """
    context = devices.get_context(device_id)
    return context.device

def get_current_device():
    "Get active device associated with the current thread"
    return current_context().device

def list_devices():
    devices.init_gpus()
    return devices.gpus

def close():
    devices.reset()

def detect():
    devlist = list_devices()
    print('Found {0} OpenCL devices'.format(len(devlist)))
    supported_count = 0
    for dev in devlist:
        attrs = []
        attrs.append(('profile', dev.profile))
        attrs.append(('type', dev.type_str))
        attrs.append(('vendor', dev.vendor))
        attrs.append(('vendor id', dev.vendor_id))

        # TODO: all supported? 
        support = '[SUPPORTED]'
        supported_count += 1

        print("device '{0}' {1}".format(dev.name, support))
        for key, val in attrs:
            print('{0:>40} {1}'.format(key, val))

    print ('Summary:\n{0} of {1} devices supported'.format(supported_count, len(devlist)))
    return supported_count > 0


@contextlib.contextmanager
def defer_cleanup():
    # compat only right now... no trashing service in opencl
    yield
