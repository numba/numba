import sys, bisect
from ctypes import *
import numpy as np
from numpy.ctypeslib import c_intp
from numbapro.cudapipeline import driver as _driver
from bitarray import bitarray
from numbapro._utils import finalizer

_pyobject_head_fields = [('pyhead1', c_size_t),
                         ('pyhead2', c_void_p),]

if hasattr(sys, 'getobjects'):
    _pyobject_head_fields = [('pyhead3', c_int),
                             ('pyhead4', c_int),] + \
                              _pyobject_head_fields

_numpy_fields = _pyobject_head_fields + \
      [('data',       c_void_p),       # data
       ('nd',         c_int),          # nd
       ('dimensions', c_void_p),       # dimensions
       ('strides',    c_void_p),       # strides
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

def ndarray_device_allocate_head(nd):
    "Allocate the metadata structure"
    gpu_head = _driver.DeviceMemory(sizeof(NumpyStructure))
    return gpu_head

def ndarray_device_allocate_data(ary):
    datasize = _driver.host_memory_size(ary)
    # allocate
    gpu_data = _driver.DeviceMemory(datasize)
    return gpu_data

def ndarray_device_transfer_data(ary, gpu_data, stream=0):
    size = _driver.host_memory_size(ary)
    # transfer data
    _driver.host_to_device(gpu_data, ary, size, stream=stream)

def ndarray_populate_head(gpu_head, gpu_data, shape, strides, stream=0):
    nd = len(shape)
    assert nd > 0

    smm = SmallMemoryManager.get()

    gpu_shape = smm.obtain(shape, stream=stream)
    gpu_strides = smm.obtain(strides, stream=stream)

    # Offset to shape and strides memory
    struct = NumpyStructure()

    # Fill the ndarray structure
    struct.nd         = nd
    struct.data       = c_void_p(_driver.device_pointer(gpu_data))
    struct.dimensions = c_void_p(_driver.device_pointer(gpu_shape))
    struct.strides    = c_void_p(_driver.device_pointer(gpu_strides))

    # transfer the memory
    _driver.host_to_device(gpu_head, struct, sizeof(struct), stream=stream)
    # gpu_head owns mm_shape and mm_strides
    _driver.device_memory_depends(gpu_head, gpu_shape, gpu_strides, gpu_data)


class SMMBucket(object):
    __slots__ = 'start', 'stop', 'refct'
    def __init__(self, start, stop, refct):
        self.start = start
        self.stop = stop
        self.refct = refct

    def overlaps(self, other):
        if self.start < other.stop and self.stop > other.start:
            return True
        elif other.start < self.stop and other.stop > self.start:
            return True
        else:
            return False

    def __lt__(self, other):
        return self.start < other.start

    def __gt__(self, other):
        return self.start > other.start

    __eq__ = NotImplemented
    __ne__ = NotImplemented

    def __repr__(self):
        return 'SMMBucket(%s, %s, %s)' % (self.start, self.stop, self.refct)

class SMMBucketCollection(object):
    def __init__(self):
        self.keys = []
        self.vals = []

    def index(self, bucket):
        "Index of the last item with a smaller bucket.start"
        return bisect.bisect_left(self.keys, bucket)

    def insert(self, bucket, value):
        i = self.index(bucket)
        self.keys.insert(i, bucket)
        self.vals.insert(i, value)

    def remove_overlap(self, bucket):
        idx = self.index(bucket)
        for i in range(idx, min(len(self) - 1, idx + 2)):
            b = self.keys[i]
            if b.overlaps(bucket):
                del self.keys[i]
                del self.vals[i]

    def __len__(self):
        return len(self.keys)


class SmallMemoryManager(object):
    '''Allocate a mapped+pinned host memory for storing small tuples of intp:
    e.g. shape and strides
        
    Default to reserve 2-MB with 32-B block.
    '''
    NBYTES = 2**21
    BLOCKSZ = 32
    __singleton = {}

    @classmethod
    def get(cls, **kws):
        '''One instance per thread.'''
        drv = _driver.Driver()
        device = drv.current_context().device
        if device not in cls.__singleton:
            cls.__singleton[device] = obj = cls(**kws)
            return obj
        else:
            return cls.__singleton[device]

    def __init__(self, nbytes=NBYTES, blocksz=BLOCKSZ):
        self.nbytes = nbytes
        self.blocksz = blocksz
        self.blockct = self.nbytes // self.blocksz
        self.dtype = np.intp
        self.itemsize = np.dtype(self.dtype).itemsize
        self.valueperblock = blocksz // self.itemsize
        assert self.blocksz * self.blockct == self.nbytes

    @_driver.require_context
    def _lazy_init(self):
        # allocate host memory
        elemct = self.nbytes // self.itemsize
        # pinned the memory for direct GPU access
        hm = _driver.HostAllocMemory(self.nbytes)
        self.hostmemory = np.ndarray(shape=elemct, dtype=self.dtype,
                                     buffer=hm)
        self.devicememory = _driver.DeviceMemory(self.nbytes)
        self.devaddrbase = _driver.device_pointer(self.devicememory)
        # initialize valuemap
        self.valuemap = {} # stores tuples -> bucket
        # initialize bucketmap
        self.bucketmap = SMMBucketCollection()  # stores bucket -> tuples
        # initialize usemap
        self.usemap = bitarray(self.blockct)
        self.usemap[:] = False
        self.usemap_last = 0

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
            pattern = bitarray('0' * nval)
            iterator = self.usemap[self.usemap_last:].itersearch(pattern)
            try: # get the first empty slot
                index = iterator.next()
            except StopIteration:
                raise Exception("Insufficient resources")
            else:
                # calc indices
                index += self.usemap_last
                start = index * self.valueperblock
                stop = start + nval
                blkstart = index
                blkstop = index + self.calc_block_use(nval)
                self.usemap_last = blkstop % self.blockct
                # put in host memory
                self.hostmemory[start:stop] = np.array(value, dtype=self.dtype)
                # new bucket
                bucket = SMMBucket(blkstart, blkstop, 1)
                # garbage collect bucket
                self.bucketmap.remove_overlap(bucket)
                # store bucket
                self.valuemap[value] = bucket
                self.bucketmap.insert(bucket, value)
                # mark use
                self.usemap[blkstart:blkstop] = True
                # H->D
                offset = index * self.blocksz
                src = self.hostmemory[start:stop]
                size = nval * self.itemsize
                ptr = self.devaddrbase + bucket.start * self.blocksz
                mp = ManagedPointer(self, value, _driver.cu_device_ptr(ptr))
                _driver.host_to_device(mp, src, size, stream=stream)
                return mp
        else:
            # already allocated, reuse
            bucket.refct += 1
            self.usemap[bucket.start:bucket.stop] = True
            ptr = self.devaddrbase + bucket.start * self.blocksz
            return ManagedPointer(self, value, _driver.cu_device_ptr(ptr))

    def release(self, value):
        bucket = self.valuemap[value]
        bucket.refct -= 1
        assert bucket.refct >= 0
        if bucket.refct == 0:
            self.usemap[bucket.start : bucket.stop] = False

class ManagedPointer(finalizer.OwnerMixin):
    __cuda_memory__ = True
    def __init__(self, parent, value, handle):
        self._handle = handle
        self.parent = parent
        self.value = value
        self._finalizer_track((parent, value))

    @classmethod
    def _finalize(cls, pair):
        parent, value = pair
        parent.release(value)

    @property
    def device_ctypes_pointer(self):
        return self._handle

    @property
    def driver(self):
        return _driver.Driver()
