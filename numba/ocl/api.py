"""
API for OpenCL
"""

from __future__ import print_function, absolute_import, division
from .ocldrv import oclarray
from . import ocldrv

try:
    long
except NameError:
    long = int


# Array to device migration ############################################

def to_device(ary, context_or_queue, copy=True, to=None):
    """
    to_device(ary, context_or_queue, copy=True, to=None)

    Allocate and transfer a NumPy ndarray in an OpenCL context.

    Parameters
    ----------
    ary : array_like
        The shape and datatype of 'ary' define the resulting device
        array. 'ary' may be the source of actual data if 'copy' is
        True
    context_or_queue : An OpenCL context or queue
        The array will be allocated in that context (or the queue's
        context if it is a queue).
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
        pass

    return devarray
