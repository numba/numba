import sys
from ctypes import *
import threading
import numpy as np
from numpy.ctypeslib import c_intp
from numbapro._cuda import driver as _cuda
from numbapro._utils.ndarray import *
from bitarray import bitarray
from numbapro._utils import finalizer

_pyobject_head_fields = [('pyhead1', c_size_t),
                         ('pyhead2', c_void_p),]

if hasattr(sys, 'getobjects'):
    _pyobject_head_fields = [('pyhead3', c_int),
                             ('pyhead4', c_int),] + \
                              _pyobject_head_fields

_numpy_fields = _pyobject_head_fields + \
      [('data', c_void_p),                      # data
       ('nd',   c_int),                         # nd
       ('dimensions', POINTER(c_intp)),       # dimensions
       ('strides', POINTER(c_intp)),          # strides
        #  NOTE: The following fields are unused.
        #        Not sending to GPU to save transfer bandwidth.
        #       ('base', c_void_p),                      # base
        #       ('desc', c_void_p),                      # descr
        #       ('flags', c_int),                        # flags
        #       ('weakreflist', c_void_p),               # weakreflist
        #       ('maskna_dtype', c_void_p),              # maskna_dtype
        #       ('maskna_data', c_void_p),               # maskna_data
        #       ('masna_strides', POINTER(c_intp)),    # masna_strides
      ]


class NumpyStructure(Structure):
    _fields_ = _numpy_fields

def ndarray_device_allocate_struct(nd):
    gpu_struct = _cuda.AllocatedDeviceMemory(sizeof(NumpyStructure))
    return gpu_struct

def ndarray_device_allocate_data(ary):
    datasize = ndarray_datasize(ary)
    # allocate
    gpu_data = _cuda.AllocatedDeviceMemory(datasize)
    return gpu_data

def ndarray_device_transfer_data(ary, gpu_data, stream=0):
    datapointer = ary.ctypes.data
    datasize = ndarray_datasize(ary)
    # transfer data
    gpu_data.to_device_raw(datapointer, datasize, stream=stream)

def ndarray_populate_struct(gpu_struct, gpu_data, shape, strides, stream=0):
    nd = len(shape)

    to_intp_p = lambda x: cast(c_void_p(x), POINTER(c_intp))

    smm = SmallMemoryManager.get()

    gpu_shape = smm.obtain(shape, stream=stream)
    gpu_strides = smm.obtain(strides, stream=stream)

    # Offset to shape and strides memory
    struct = NumpyStructure()

    # Fill the ndarray structure
    struct.nd = nd
    struct.data = c_void_p(gpu_data._handle.value)
    struct.dimensions = to_intp_p(gpu_shape._handle.value)
    struct.strides = to_intp_p(gpu_strides._handle.value)

    # transfer the memory
    gpu_struct.to_device_raw(addressof(struct), sizeof(struct), stream=stream)
    # gpu_struct owns mm_shape and mm_strides
    gpu_struct.add_dependencies(gpu_shape, gpu_strides)


class SMMBucket(object):
    __slots__ = 'start', 'stop', 'refct'
    def __init__(self, start, stop, refct):
        self.start = start
        self.stop = stop
        self.refct = refct

class SmallMemoryManager(object):
    '''Allocate a mapped+pinned host memory for storing small tuples of intp:
    e.g. shape and strides
        
    Default to reserve 4-MB with 16-B block.
    '''
    NBYTES = 2**22
    BLOCKSZ = 16

    Driver = _cuda.Driver()

    @classmethod
    def get(cls, *args, **kws):
        '''One instance per thread.'''
        if not hasattr(cls.Driver._THREAD_LOCAL, 'smm'):
            cls.Driver._THREAD_LOCAL.smm = cls(*args, **kws)
        return cls.Driver._THREAD_LOCAL.smm

    def __init__(self, nbytes=NBYTES, blocksz=BLOCKSZ):
        self.nbytes = nbytes
        self.blocksz = blocksz
        self.blockct = self.nbytes // self.blocksz
        self.dtype = np.intp
        self.itemsize = np.dtype(self.dtype).itemsize
        self.valueperblock = blocksz // self.itemsize
        assert self.blocksz * self.blockct == self.nbytes

    def _lazy_init(self):
        import numbapro._cuda.default  # ensure we have a context
        # allocate host memory
        elemct = self.nbytes // self.itemsize
        hm = self.hostmemory = np.empty(elemct, dtype=self.dtype)
        # pinned the memory for direct GPU access
        self.pinnedmemory = _cuda.PinnedMemory(hm.ctypes.data, self.nbytes)
        # allocate gpu memory
        self.devicememory = _cuda.AllocatedDeviceMemory(self.nbytes)
        self.devaddrbase = self.devicememory._handle.value
        # initialize valuemap
        self.valuemap = {} # stores tuples -> (start, stop, refct)
        # initialize usemap
        self.usemap = bitarray(self.blockct)
        self.usemap[:] = False

    def calc_block_use(self, n):
        from math import ceil
        out = int(ceil(float(n) / self.valueperblock))
        return out

    def obtain(self, value, stream=0):
        assert isinstance(value, tuple), type(value)
        if not hasattr(self, 'devicememory'):
            self._lazy_init()
        try:
            bucket = self.valuemap[value]
        except KeyError:
            # perform allocation
            ## find location
            nval = len(value)
            iterator = self.usemap.itersearch(bitarray('0' * nval))
            try: # get the first empty slot
                index = iterator.next()
            except StopIteration:
                raise Exception("Insufficient resources")
            else:
                # calc indices
                start = index * self.valueperblock
                stop = start + nval
                blkstart = index
                blkstop = index + self.calc_block_use(nval)
                # put in host memory
                self.hostmemory[start:stop] = np.array(value, dtype=self.dtype)
                # store bucket
                self.valuemap[value] = bucket = SMMBucket(blkstart, blkstop, 1)
                # mark use
                self.usemap[blkstart:blkstop] = True
                # H->D
                offset = index * self.blocksz
                src = self.hostmemory.ctypes.data + offset
                size = nval * self.itemsize
                self.devicememory.to_device_raw(src, size, stream=stream,
                                                offset=offset)
                ptr = self.devaddrbase + bucket.start * self.blocksz
                return ManagedPointer(self, value, _cuda.cu_device_ptr(ptr))
        else:
            # already allocated
            bucket.refct += 1
            ptr = self.devaddrbase + bucket.start * self.blocksz
            return ManagedPointer(self, value, _cuda.cu_device_ptr(ptr))

    def release(self, value):
        bucket = self.valuemap[value]
        bucket.refct -= 1
        assert bucket.refct >= 0
        if bucket.refct == 0:
            # assert all(self.usemap[bucket.start:bucket.stop])
            del self.valuemap[value]
            self.usemap[bucket.start : bucket.stop] = False


class ManagedPointer(_cuda.DevicePointer, finalizer.OwnerMixin):
    def __init__(self, parent, value, handle):
        _cuda.DevicePointer.__init__(self, handle)
        self.parent = parent
        self.value = value
        self._finalizer_track((parent, value))

    @classmethod
    def _finalize(cls, pair):
        parent, value = pair
        parent.release(value)

