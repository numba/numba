from __future__ import print_function, absolute_import, division
import numpy as np
import numba.ctypes_support as ctypes
import contextlib
from collections import deque
from . import devices, driver
from numba.targets.arrayobj import make_array_ctype
from numba.targets.registry import CPUTarget


def _calc_array_sizeof(ndim):
    """Use the ABI size in the CPU target
    """
    ctx = CPUTarget.target_context
    return ctx.calc_array_sizeof(ndim)


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
    # The number of preallocated array head
    maxsize = 2 ** 10

    # Maximum size for each array head
    #    = 4 (ndim) * 8 (sizeof intp) * 2 (shape strides) + 8 (ptr)
    elemsize = _calc_array_sizeof(4)

    # Number of page-locked staging area
    num_stages = 5

    def __new__(cls, context):
        mm = context.extras.get(cls)
        if mm is None:
            mm = object.__new__(cls)
            mm.init(context)
            context.extras[cls] = mm
        return mm

    def init(self, context):
        self.context = context
        self.data = self.context.memalloc(self.elemsize * self.maxsize)
        self.queue = deque()
        for i in range(self.maxsize):
            offset = i * self.elemsize
            mem = self.data.view(offset, offset + self.elemsize)
            self.queue.append(mem)
        self.allocated = set()
        # A staging buffer to temporary store data for copying
        self.stages = [np.ndarray(shape=self.elemsize, dtype=np.byte,
                                  order='C',
                                  buffer=context.memhostalloc(self.elemsize))
                       for _ in range(self.num_stages)]
        self.events = [context.create_event(timing=False)
                       for _ in range(self.num_stages)]

        self.stage_queue = deque(zip(self.events, self.stages))

    @contextlib.contextmanager
    def get_stage(self, stream):
        """Get a pagelocked staging area and record the event when we are done.
        """
        evt, stage = self.stage_queue.popleft()
        if not evt.query():
            evt.wait(stream=stream)
        yield stage
        evt.record(stream=stream)
        self.stage_queue.append((evt, stage))

    def allocate(self, nd):
        arraytype = make_array_ctype(nd)
        sizeof = ctypes.sizeof(arraytype)
        # Oversized or insufficient space
        if sizeof > self.elemsize or not self.queue:
            return _allocate_head(nd)

        mem = self.queue.popleft()
        self.allocated.add(mem)
        return mem

    def free(self, mem):
        if mem in self.allocated:
            self.allocated.discard(mem)
            self.queue.append(mem)

    def write(self, data, to, stream=0):
        if data.size > self.elemsize:
            # Cannot use pinned staging memory
            stage = data
            driver.host_to_device(to, stage, data.size, stream=stream)
        else:
            # Can use pinned staging memory
            with self.get_stage(stream=stream) as stage_area:
                stage = stage_area[:data.size]
                stage[:] = data
                driver.host_to_device(to, stage, data.size, stream=stream)

    def __repr__(self):
        return "<cuda managed memory %s >" % (self.context.device,)


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


def ndarray_populate_head(gpu_mem, gpu_data, shape, strides, stream=0):
    """
    Populate the array header
    """
    nd = len(shape)
    assert nd > 0, "0 or negative dimension"

    arraytype = make_array_ctype(nd)
    struct = arraytype(parent=None,
                       data=driver.device_pointer(gpu_data),
                       shape=shape,
                       strides=strides)

    gpu_head = gpu_mem.allocate(nd)
    databytes = np.ndarray(shape=ctypes.sizeof(struct), dtype=np.byte,
                           buffer=struct)
    gpu_mem.write(databytes, gpu_head, stream=stream)
    driver.device_memory_depends(gpu_head, gpu_data)
    return gpu_head
