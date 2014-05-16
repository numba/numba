"""
OpenCL driver implementation

device memory related functions, providing the tools to deal with objects
that represent memory in the OpenCL context.
"""

from __future__ import absolute_import, print_function, division

from ... import mviewbuf


try:
    long
except NameError:
    long = int


def is_device_memory(obj):
    """All ocl memory objects will be recognized by having the attribute
    '__ocl_memory__' defined and evaluate to True
    """
    return getattr(obj, '__ocl_memory__', False)

def require_device_memory(obj):
    if not is_device_memory(obj):
        raise Exception("Not an OpenCL memory object.")


def host_pointer(obj):
    """
    This is the same function as cuda driver host_pointer.
    Common functions should be merged together.
    memory management should be similar in both cases, where possible.
    """
    if isinstance(obj, (int, long)):
        return obj
    return mviewbuf.memoryview_get_buffer(obj)

def host_memory_size(obj):
    s, e = host_memory_extents(obj)
    assert e >= s, "memory extent of negative size"
    return e-s

def host_memory_extents(obj):
    return mviewbuf.memoryview_get_extents(obj)
