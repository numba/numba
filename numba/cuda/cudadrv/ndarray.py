from __future__ import print_function, absolute_import, division
import numba.ctypes_support as ctypes
from . import devices, driver


class ArrayHeaderManager(object):
    """
    Manages array header memory for reusing the allocation.

    It allocates one big chunk of memory and partition it for fix sized array
    header.  It currently stores up to 4D array header in 64-bit mode or 8D
    array header in 32-bit mode.

    This allows the small array header allocation to be reused to avoid
    breaking asynchronous streams and avoid fragmentation of memory.

    When run out of preallocated space, it automatically fallback to regular
    allocation.
    """

    # Caches associated contexts
    #    There is one array header manager per context.
    context_map = {}

    # The number of preallocated array head
    maxsize = 2 ** 10

    # Maximum size for each array head
    #    = 4 (ndim) * 8 (sizeof intp) * 2 (shape strides) + 8 (ptr)
    elemsize = 72

    def __new__(cls, context):
        key = context.handle.value
        mm = cls.context_map.get(key)
        if mm is None:
            mm = object.__new__(cls)
            mm.init(context)
            cls.context_map[key] = mm

        return mm

    def init(self, context):
        self.context = context
        self.data = self.context.memalloc(self.elemsize * self.maxsize)
        self.queue = []
        for i in range(self.maxsize):
            offset = i * self.elemsize
            mem = self.data.view(offset, offset + self.elemsize)
            self.queue.append(mem)
        self.allocated = set()

    def allocate(self, nd):
        arraytype = make_array_ctype(nd)
        sizeof = ctypes.sizeof(arraytype)

        # Oversized or insufficient space
        if sizeof >= self.elemsize or not self.queue:
            return _allocate_head(nd)

        mem = self.queue.pop()
        self.allocated.add(mem)
        return mem

    def free(self, mem):
        if mem in self.allocated:
            self.allocated.discard(mem)
            self.queue.append(mem)

    def __repr__(self):
        return "<cuda managed memory %s >" % (self.context.device,)


def make_array_ctype(ndim):
    """Create a array header type for a given dimension.
    """
    c_intp = ctypes.c_ssize_t

    class c_array(ctypes.Structure):
        _fields_ = [('data', ctypes.c_void_p),
                    ('shape', c_intp * ndim),
                    ('strides', c_intp * ndim)]

    return c_array


def _allocate_head(nd):
    """Allocate the metadata structure
    """
    arraytype = make_array_ctype(nd)
    gpu_head = devices.get_context().memalloc(ctypes.sizeof(arraytype))
    return gpu_head


def ndarray_device_allocate_data(ary):
    """
    Allocate gpu data buffer
    """
    datasize = driver.host_memory_size(ary)
    # allocate
    gpu_data = devices.get_context().memalloc(datasize)
    return gpu_data


def ndarray_populate_head(gpu_head, gpu_data, shape, strides, stream=0):
    """
    Populate the array header
    """
    nd = len(shape)
    assert nd > 0, "0 or negative dimension"

    arraytype = make_array_ctype(nd)
    struct = arraytype(data=driver.device_pointer(gpu_data),
                       shape=shape,
                       strides=strides)

    driver.host_to_device(gpu_head, struct, ctypes.sizeof(struct),
                          stream=stream)
    driver.device_memory_depends(gpu_head, gpu_data)
