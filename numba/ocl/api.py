"""
API for OpenCL
"""

from __future__ import print_function, absolute_import, division
from .ocldrv import oclarray
from . import ocldrv
from .ocldrv import oclarray
from .. import mviewbuf
import numpy as np

try:
    long
except NameError:
    long = int


# Array to device migration ############################################

def to_device(context_or_queue, ary, copy=True, to=None):
    """
    to_device(ary, context_or_queue, copy=True, to=None)

    Allocate and transfer a NumPy ndarray in an OpenCL context.

    Parameters
    ----------
    context_or_queue : An OpenCL context or queue
        The array will be allocated in that context (or the queue's
        context if it is a queue).
    ary : array_like
        The shape and datatype of 'ary' define the resulting device
        array. 'ary' may be the source of actual data if 'copy' is
        True
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
    assert(isinstance(context_or_queue, (ocldrv.Context, ocldrv.CommandQueue)))
    if isinstance(context_or_queue, ocldrv.CommandQueue):
        ctxt = context_or_queue.context
        q = context_or_queue
    else:
        ctxt = context_or_queue
        q = None

    if to is None:
        if copy:
            devarray = oclarray.from_array(ary, ctxt)
        else:
            devarray = oclarray.from_array_like(ary, ctxt)
    else:
        oclarray.require_ocl_ndarray(to)
        devarray = to
        if copy:
            devarray.copy_to_device(ary, q)
        else:
            pass # does this option make sense?


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
