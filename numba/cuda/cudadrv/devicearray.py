"""
A CUDA ND Array is recognized by checking the __cuda_memory__ attribute
on the object.  If it exists and evaluate to True, it must define shape,
strides, dtype and size attributes similar to a NumPy ndarray.
"""
from __future__ import print_function, absolute_import, division
import warnings
import math
import operator
import numpy as np
from functools import reduce
from .ndarray import ndarray_device_allocate_head, ndarray_populate_head
from . import driver as _driver
from . import devices

try:
    long
except NameError:
    long = int


def is_cuda_ndarray(obj):
    "Check if an object is a CUDA ndarray"
    return getattr(obj, '__cuda_ndarray__', False)


def verify_cuda_ndarray_interface(obj):
    "Verify the CUDA ndarray interface for an obj"
    require_cuda_ndarray(obj)

    def requires_attr(attr, typ):
        if not hasattr(obj, attr):
            raise AttributeError(attr)
        if not isinstance(getattr(obj, attr), typ):
            raise AttributeError('%s must be of type %s' % (attr, typ))

    requires_attr('shape', tuple)
    requires_attr('strides', tuple)
    requires_attr('dtype', np.dtype)
    requires_attr('size', (int, long))


def require_cuda_ndarray(obj):
    "Raises ValueError is is_cuda_ndarray(obj) evaluates False"
    if not is_cuda_ndarray(obj):
        raise ValueError('require an cuda ndarray object')


class DeviceNDArrayBase(object):
    """A on GPU NDArray representation
    """
    __cuda_memory__ = True
    __cuda_ndarray__ = True # There must be a gpu_head and gpu_data attribute

    def __init__(self, shape, strides, dtype, stream=0, writeback=None,
                 gpu_head=None, gpu_data=None):
        """
        Arguments

        shape: array shape.
        strides: array strides.
        dtype: data type as numpy.dtype.
        stream: cuda stream.
        writeback: Deprecated.
        gpu_head: user provided device memory for the ndarray head structure
        gpu_data: user provided device memory for the ndarray data buffer
        """
        if isinstance(shape, (int, long)):
            shape = (shape,)
        if isinstance(strides, (int, long)):
            strides = (strides,)
        self.ndim = len(shape)
        if len(strides) != self.ndim:
            raise ValueError('strides not match ndim')
        self.shape = tuple(shape)
        self.strides = tuple(strides)
        self.dtype = np.dtype(dtype)
        self.size = int(np.prod(self.shape))
        # prepare gpu memory
        if gpu_data is None:
            self.alloc_size = _driver.memory_size_from_info(self.shape,
                                                            self.strides,
                                                            self.dtype.itemsize)
            gpu_data = devices.get_context().memalloc(self.alloc_size)
        else:
            self.alloc_size = _driver.device_memory_size(gpu_data)

        if gpu_head is None:
            gpu_head = ndarray_device_allocate_head(self.ndim)
            ndarray_populate_head(gpu_head, gpu_data, self.shape,
                                  self.strides, stream=stream)
        self.gpu_head = gpu_head
        self.gpu_data = gpu_data

        self.__writeback = writeback # should deprecate the use of this

        # define the array interface to work with numpy
        #
        # XXX: problem with data being accessed.
        #      is NULL pointer alright?
        #
        #        self.__array_interface__ = {
        #            'shape'     : self.shape,
        #            'typestr'   : self.dtype.str,
        #            'data'      : (0, True),
        #            'version'   : 3,
        #        }


    @property
    def device_ctypes_pointer(self):
        "Returns the ctypes pointer to the GPU data buffer"
        return self.gpu_data.device_ctypes_pointer

    def copy_to_device(self, ary, stream=0):
        """Copy `ary` to `self`.

        If `ary` is a CUDA memory, perform a device-to-device transfer.
        Otherwise, perform a a host-to-device transfer.
        """
        if _driver.is_device_memory(ary):
            sz = min(_driver.device_memory_size(self),
                     _driver.device_memory_size(ary))
            _driver.device_to_device(self, ary, sz, stream=stream)
        else:
            sz = min(_driver.host_memory_size(ary), self.alloc_size)
            _driver.host_to_device(self, ary, sz, stream=stream)

    def copy_to_host(self, ary=None, stream=0):
        """Copy ``self`` to ``ary`` or create a new numpy ndarray
        if ``ary`` is ``None``.

        Always returns the host array.
        """
        if ary is None:
            hostary = np.empty(shape=self.alloc_size, dtype=np.byte)
        else:
            if ary.dtype != self.dtype:
                raise TypeError('incompatible dtype')

            if ary.shape != self.shape:
                scalshapes = (), (1,)
                if not (ary.shape in scalshapes and self.shape in scalshapes):
                    raise TypeError('incompatible shape; device %s; host %s' %
                                    (self.shape, ary.shape))
            if ary.strides != self.strides:
                scalstrides = (), (self.dtype.itemsize,)
                if not (ary.strides in scalstrides and
                                self.strides in scalstrides):
                    raise TypeError('incompatible strides; device %s; host %s' %
                                    (self.strides, ary.strides))
            hostary = ary
        _driver.device_to_host(hostary, self, self.alloc_size, stream=stream)

        if ary is None:
            hostary = np.ndarray(shape=self.shape, strides=self.strides,
                                 dtype=self.dtype, buffer=hostary)
        return hostary

    def to_host(self, stream=0):
        warnings.warn("to_host() is deprecated and will be removed",
                      DeprecationWarning)
        if self.__writeback is None:
            raise ValueError("no associated writeback array")
        self.copy_to_host(self.__writeback, stream=stream)

    def split(self, section, stream=0):
        '''Split the array into equal partition of the `section` size.
        If the array cannot be equally divided, the last section will be
        smaller.
        '''
        if self.ndim != 1:
            raise ValueError("only support 1d array")
        if self.strides[0] != self.dtype.itemsize:
            raise ValueError("only support unit stride")
        nsect = int(math.ceil(float(self.size) / section))
        strides = self.strides
        itemsize = self.dtype.itemsize
        for i in range(nsect):
            begin = i * section
            end = min(begin + section, self.size)
            shape = (end - begin,)
            gpu_data = self.gpu_data.view(begin * itemsize, end * itemsize)
            gpu_head = ndarray_device_allocate_head(1)
            ndarray_populate_head(gpu_head, gpu_data, shape, strides,
                                  stream=stream)
            yield DeviceNDArray(shape, strides, dtype=self.dtype, stream=stream,
                                gpu_head=gpu_head, gpu_data=gpu_data)

    def as_cuda_arg(self):
        """Returns a device memory object that is used as the argument.
        """
        return self.gpu_head


class DeviceNDArray(DeviceNDArrayBase):
    def _yields_f_strides_by_shape(self, shape=None):
        """yields the f-contigous strides
        """
        shape = self.shape if shape is None else shape
        itemsize = self.dtype.itemsize
        yield itemsize
        sum = 1
        for s in shape[:-1]:
            sum *= s
            yield sum * itemsize

    def _yields_c_strides_by_shape(self, shape=None):
        '''yields the c-contigous strides in
        '''
        shape = self.shape if shape is None else shape
        itemsize = self.dtype.itemsize

        def gen():
            yield itemsize
            sum = 1
            for s in reversed(shape[1:]):
                sum *= s
                yield sum * itemsize

        for i in reversed(list(gen())):
            yield i

    def is_f_contigous(self):
        return all(i == j for i, j in
                   zip(self._yields_f_strides_by_shape(), self.strides))

    def is_c_contigous(self):
        return all(i == j for i, j in
                   zip(self._yields_c_strides_by_shape(),
                       self.strides))

    def reshape(self, *newshape, **kws):
        '''reshape(self, *newshape, order='C'):

        Reshape the array and keeping the original data
        '''
        order = kws.pop('order', 'C')
        if kws:
            raise TypeError('unknown keyword arguments %s' % kws.keys())
        if order not in 'CFA':
            raise ValueError('order not C|F|A')
            # compute new array size
        if len(newshape) > 1:
            newsize = reduce(operator.mul, newshape)
        else:
            (newsize,) = newshape
        if newsize != self.size:
            raise ValueError("reshape changes the size of the array")
        elif self.is_f_contigous() or self.is_c_contigous():
            if order == 'A':
                order = 'F' if self.is_f_contigous() else 'C'
            if order == 'C':
                newstrides = list(self._yields_c_strides_by_shape(newshape))
            elif order == 'F':
                newstrides = list(self._yields_f_strides_by_shape(newshape))
            else:
                assert False, 'unreachable'

            ret = DeviceNDArray(shape=newshape, strides=newstrides,
                                dtype=self.dtype, gpu_data=self.gpu_data)

            return ret
        else:
            raise NotImplementedError("reshaping non-contiguous array requires"
                                      "autojitting a special kernel to complete")

    def ravel(self, order='C', stream=0):
        if order not in 'CFA':
            raise ValueError('order not C|F|A')
        elif self.ndim == 1:
            return DeviceNDArray(shape=self.shape, strides=self.strides,
                                 dtype=self.dtype, gpu_data=self.gpu_data)
        elif (order == 'C' and self.is_c_contigous() or
                          order == 'F' and self.is_f_contigous()):
            # just flatten it
            return DeviceNDArray(shape=self.size, strides=self.dtype.itemsize,
                                 dtype=self.dtype, gpu_data=self.gpu_data,
                                 stream=stream)
        else:
            raise NotImplementedError("this ravel operation requires "
                                      "autojitting a special kernel to complete")

    def __getitem__(self, index):
        def is_tuple_of_ints(x):
            if isinstance(x, tuple):
                return all(isinstance(i, int) for i in x)
            return False

        if isinstance(index, int):
            return self._getitem_index(index)
        elif is_tuple_of_ints(index):
            return self._getitem_index(*index)
        elif isinstance(index, tuple):
            return self._getitem_slice(*index)
        else:
            return self._getitem_slice(index)

    def _getitem_index(self, *indices):
        if len(indices) < self.ndim:
            self._getitem_slice(*indices)
        if len(indices) > self.ndim:
            raise TypeError("too many indices")
        indices = _normalize_indices(indices, self.shape)
        offset = (np.array(indices) * np.array(self.strides)).sum()
        itemsize = self.dtype.itemsize
        new_data = self.gpu_data.view(offset, offset + itemsize)
        hostary = np.empty(1, dtype=self.dtype)
        _driver.device_to_host(dst=hostary, src=new_data, size=itemsize)
        return hostary[0]

    def _getitem_slice(self, *slices):
        slices = [_make_slice(sl) for sl in slices]
        slices = _normalize_slices(slices, self.shape)
        bounds = [_compute_slice_offsets(sl.start, sl.stop, shp)
                  for sl, shp in zip(slices, self.shape)]
        indices = [s for s, e in bounds]
        lastindices = [e - 1 for s, e in bounds]
        itemsize = self.dtype.itemsize
        strides_arr = np.array(self.strides)
        offset = (np.array(indices) * strides_arr).sum()
        endoffset = (np.array(lastindices) * strides_arr).sum() + itemsize
        shapes = [e - s for s, e in bounds]
        strides = [(st if sl.step is None else sl.step)
                   for sl, st in zip(slices, self.strides)]
        new_data = self.gpu_data.view(offset, endoffset)
        arr = type(self)(shape=shapes, strides=strides, dtype=self.dtype,
                         gpu_data=new_data)
        return arr


def _compute_slice_offsets(start, stop, shape):
    assert start is not None or stop is not None

    if start is None:
        start = 0
    if stop is None:
        stop = shape
    return start, stop


def _normalize_indices(indices, shape):
    return [_normalize_index(i, s) for i, s in zip(indices, shape)]


def _normalize_index(index, shape):
    index = (shape + index if index < 0 else index)
    if index >= shape:
        raise IndexError("out of bound")
    return index


def _normalize_bound(index, shape):
    index = (shape + index if index < 0 else index)
    return min(index, shape)


def _normalize_slices(slices, shape):
    out = []
    for sl, sh in zip(slices, shape):
        start, stop, step = sl.start, sl.stop, sl.step
        if start is not None:
            start = _normalize_bound(start, sh)
        if stop is not None:
            stop = _normalize_bound(stop, sh)
        out.append(slice(start, stop, step))
    return out


def _make_slice(val):
    if isinstance(val, slice):
        return val
    return slice(val, val + 1)


class MappedNDArray(DeviceNDArrayBase, np.ndarray):
    def device_setup(self, gpu_data, stream=0):
        gpu_head = ndarray_device_allocate_head(self.ndim)

        ndarray_populate_head(gpu_head, gpu_data, self.shape,
                              self.strides, stream=stream)

        self.gpu_data = gpu_data
        self.gpu_head = gpu_head


def from_array_like(ary, stream=0, gpu_head=None, gpu_data=None):
    "Create a DeviceNDArray object that is like ary."
    if ary.ndim == 0:
        ary = ary.reshape(1)
    return DeviceNDArray(ary.shape, ary.strides, ary.dtype,
                         writeback=ary, stream=stream, gpu_head=gpu_head,
                         gpu_data=gpu_data)


def auto_device(ary, stream=0, copy=True):
    if _driver.is_device_memory(ary):
        return ary, False
    else:
        if not ary.flags['C_CONTIGUOUS'] and not ary.flags['F_CONTIGUOUS']:
            if ary.ndim != 1 or ary.shape[0] != 1 or ary.strides[0] != 0:
                raise ValueError(
                    "Array contains non-contiguous buffer and cannot "
                    "be transferred as a single memory region.  "
                    "Please ensure contiguous buffer with numpy"
                    ".ascontiguousarray()")
        devarray = from_array_like(ary, stream=stream)
        if copy:
            devarray.copy_to_device(ary, stream=stream)
        return devarray, True

