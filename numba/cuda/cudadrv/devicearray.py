"""
A CUDA ND Array is recognized by checking the __cuda_memory__ attribute
on the object.  If it exists and evaluate to True, it must define shape,
strides, dtype and size attributes similar to a NumPy ndarray.
"""
from __future__ import print_function, absolute_import, division
import warnings
import math
import numpy as np
from .ndarray import (ndarray_populate_head, ArrayHeaderManager)
from . import driver as _driver
from . import devices
from numba import dummyarray, types, numpy_support

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
        Args
        ----

        shape
            array shape.
        strides
            array strides.
        dtype
            data type as numpy.dtype.
        stream
            cuda stream.
        writeback
            Deprecated.
        gpu_head
            user provided device memory for the ndarray head structure
        gpu_data
            user provided device memory for the ndarray data buffer
        """
        if isinstance(shape, (int, long)):
            shape = (shape,)
        if isinstance(strides, (int, long)):
            strides = (strides,)
        self.ndim = len(shape)
        if len(strides) != self.ndim:
            raise ValueError('strides not match ndim')
        self._dummy = dummyarray.Array.from_desc(0, shape, strides,
                                                 dtype.itemsize)
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

        self.gpu_mem = ArrayHeaderManager(devices.get_context())

        if gpu_head is None:
            gpu_head = self.gpu_mem.allocate(self.ndim)
            ndarray_populate_head(gpu_head, gpu_data, self.shape,
                                  self.strides, stream=stream)
        self.gpu_head = gpu_head
        self.gpu_data = gpu_data

        self.__writeback = writeback # should deprecate the use of this

    def __del__(self):
        try:
            self.gpu_mem.free(self.gpu_head)
        except:
            pass

    @property
    def _numba_type_(self):
        """
        Magic attribute expected by Numba to get the numba type that
        represents this object.
        """
        dtype = numpy_support.from_dtype(self.dtype)
        return types.Array(dtype, self.ndim, 'A')

    @property
    def device_ctypes_pointer(self):
        """Returns the ctypes pointer to the GPU data buffer
        """
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
        """Split the array into equal partition of the `section` size.
        If the array cannot be equally divided, the last section will be
        smaller.
        """
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
            yield DeviceNDArray(shape, strides, dtype=self.dtype, stream=stream,
                                gpu_data=gpu_data)

    def as_cuda_arg(self):
        """Returns a device memory object that is used as the argument.
        """
        return self.gpu_head


class DeviceNDArray(DeviceNDArrayBase):
    def is_f_contiguous(self):
        return self._dummy.is_f_contig

    def is_c_contiguous(self):
        return self._dummy.is_c_contig

    def reshape(self, *newshape, **kws):
        """reshape(self, *newshape, order='C'):

        Reshape the array and keeping the original data
        """
        if len(newshape) == 1 and isinstance(newshape, (tuple, list)):
            newshape = newshape[0]

        cls = type(self)
        if newshape == self.shape:
            # nothing to do
            return cls(shape=self.shape, strides=self.strides,
                       dtype=self.dtype, gpu_data=self.gpu_data)

        newarr, extents = self._dummy.reshape(*newshape, **kws)

        if extents == [self._dummy.extent]:
            return cls(shape=newarr.shape, strides=newarr.strides,
                       dtype=self.dtype, gpu_data=self.gpu_data)
        else:
            raise NotImplementedError("operation requires copying")

    def ravel(self, order='C', stream=0):
        cls = type(self)
        newarr, extents = self._dummy.ravel(order=order)

        if extents == [self._dummy.extent]:
            return cls(shape=newarr.shape, strides=newarr.strides,
                       dtype=self.dtype, gpu_data=self.gpu_data,
                       gpu_head=self.gpu_head, stream=stream)

        else:
            raise NotImplementedError("operation requires copying")

    def __getitem__(self, item):
        arr = self._dummy.__getitem__(item)
        extents = list(arr.iter_contiguous_extent())
        cls = type(self)
        if len(extents) == 1:
            newdata = self.gpu_data.view(*extents[0])

            if dummyarray.is_element_indexing(item, self.ndim):
                hostary = np.empty(1, dtype=self.dtype)
                _driver.device_to_host(dst=hostary, src=newdata,
                                       size=self._dummy.itemsize)
                return hostary[0]
            else:
                return cls(shape=arr.shape, strides=arr.strides,
                           dtype=self.dtype, gpu_data=newdata)
        else:
            newdata = self.gpu_data.view(*arr.extent)
            return cls(shape=arr.shape, strides=arr.strides,
                       dtype=self.dtype, gpu_data=newdata)


class MappedNDArray(DeviceNDArrayBase, np.ndarray):
    """
    A host array that uses CUDA mapped memory.
    """

    def device_setup(self, gpu_data, stream=0):
        self.gpu_mem = ArrayHeaderManager(devices.get_context())

        gpu_head = self.gpu_mem.allocate(self.ndim)
        ndarray_populate_head(gpu_head, gpu_data, self.shape,
                              self.strides, stream=stream)

        self.gpu_data = gpu_data
        self.gpu_head = gpu_head

    def __del__(self):
        try:
            self.gpu_mem.free(self.gpu_head)
        except:
            pass


def from_array_like(ary, stream=0, gpu_head=None, gpu_data=None):
    "Create a DeviceNDArray object that is like ary."
    if ary.ndim == 0:
        ary = ary.reshape(1)
    return DeviceNDArray(ary.shape, ary.strides, ary.dtype,
                         writeback=ary, stream=stream, gpu_head=gpu_head,
                         gpu_data=gpu_data)


errmsg_contiguous_buffer = ("Array contains non-contiguous buffer and cannot "
                            "be transferred as a single memory region. Please "
                            "ensure contiguous buffer with numpy "
                            ".ascontiguousarray()")


def sentry_contiguous(ary):
    if not ary.flags['C_CONTIGUOUS'] and not ary.flags['F_CONTIGUOUS']:
        if ary.ndim != 1 or ary.shape[0] != 1 or ary.strides[0] != 0:
            raise ValueError(errmsg_contiguous_buffer)


def auto_device(ary, stream=0, copy=True):
    if _driver.is_device_memory(ary):
        return ary, False
    else:
        sentry_contiguous(ary)
        devarray = from_array_like(ary, stream=stream)
        if copy:
            devarray.copy_to_device(ary, stream=stream)
        return devarray, True

