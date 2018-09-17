"""
A CUDA ND Array is recognized by checking the __cuda_memory__ attribute
on the object.  If it exists and evaluate to True, it must define shape,
strides, dtype and size attributes similar to a NumPy ndarray.
"""
from __future__ import print_function, absolute_import, division

import warnings
import math
import functools
import copy
from numba import six
from ctypes import c_void_p

import numpy as np

import numba
from . import driver as _driver
from . import devices
from numba import dummyarray, types, numpy_support
from numba.unsafe.ndarray import to_fixed_tuple

try:
    lru_cache = getattr(functools, 'lru_cache')(None)
except AttributeError:
    # Python 3.1 or lower
    def lru_cache(func):
        return func


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
    requires_attr('size', six.integer_types)


def require_cuda_ndarray(obj):
    "Raises ValueError is is_cuda_ndarray(obj) evaluates False"
    if not is_cuda_ndarray(obj):
        raise ValueError('require an cuda ndarray object')


class DeviceNDArrayBase(object):
    """A on GPU NDArray representation
    """
    __cuda_memory__ = True
    __cuda_ndarray__ = True     # There must be gpu_data attribute

    def __init__(self, shape, strides, dtype, stream=0, writeback=None,
                 gpu_data=None):
        """
        Args
        ----

        shape
            array shape.
        strides
            array strides.
        dtype
            data type as np.dtype.
        stream
            cuda stream.
        writeback
            Deprecated.
        gpu_data
            user provided device memory for the ndarray data buffer
        """
        if isinstance(shape, six.integer_types):
            shape = (shape,)
        if isinstance(strides, six.integer_types):
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
        if self.size > 0:
            if gpu_data is None:
                self.alloc_size = _driver.memory_size_from_info(self.shape,
                                                                self.strides,
                                                                self.dtype.itemsize)
                gpu_data = devices.get_context().memalloc(self.alloc_size)
            else:
                self.alloc_size = _driver.device_memory_size(gpu_data)
        else:
            # Make NULL pointer for empty allocation
            gpu_data = _driver.MemoryPointer(context=devices.get_context(),
                                             pointer=c_void_p(0), size=0)
            self.alloc_size = 0

        self.gpu_data = gpu_data

        self.__writeback = writeback    # should deprecate the use of this
        self.stream = stream

    @property
    def __cuda_array_interface__(self):
        return {
            'shape': tuple(self.shape),
            'strides': tuple(self.strides),
            'data': (self.device_ctypes_pointer.value, False),
            'typestr': self.dtype.str,
            'version': 0,
        }

    def bind(self, stream=0):
        """Bind a CUDA stream to this object so that all subsequent operation
        on this array defaults to the given stream.
        """
        clone = copy.copy(self)
        clone.stream = stream
        return clone

    @property
    def T(self):
        return self.transpose()

    def transpose(self, axes=None):
        if axes and tuple(axes) == tuple(range(self.ndim)):
            return self
        elif self.ndim != 2:
            raise NotImplementedError("transposing a non-2D DeviceNDArray isn't supported")
        elif axes is not None and set(axes) != set(range(self.ndim)):
            raise ValueError("invalid axes list %r" % (axes,))
        else:
            from numba.cuda.kernels.transpose import transpose
            return transpose(self)

    def _default_stream(self, stream):
        return self.stream if not stream else stream

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
        if self.gpu_data is None:
            return c_void_p(0)
        else:
            return self.gpu_data.device_ctypes_pointer

    @devices.require_context
    def copy_to_device(self, ary, stream=0):
        """Copy `ary` to `self`.

        If `ary` is a CUDA memory, perform a device-to-device transfer.
        Otherwise, perform a a host-to-device transfer.
        """
        if ary.size == 0:
            # Nothing to do
            return

        sentry_contiguous(self)
        stream = self._default_stream(stream)

        if _driver.is_device_memory(ary):
            sentry_contiguous(ary)

            if self.flags['C_CONTIGUOUS'] != ary.flags['C_CONTIGUOUS']:
                raise ValueError("Can't copy %s-contiguous array to a %s-contiguous array" % (
                    'C' if ary.flags['C_CONTIGUOUS'] else 'F',
                    'C' if self.flags['C_CONTIGUOUS'] else 'F',
                ))

            sz = min(self.alloc_size, ary.alloc_size)
            _driver.device_to_device(self, ary, sz, stream=stream)
        else:
            # Ensure same contiguous-nous. Only copies (host-side)
            # if necessary (e.g. it needs to materialize a strided view)
            ary = np.array(
                ary,
                order='C' if self.flags['C_CONTIGUOUS'] else 'F',
                subok=True,
                copy=False)

            sz = min(_driver.host_memory_size(ary), self.alloc_size)
            _driver.host_to_device(self, ary, sz, stream=stream)

    @devices.require_context
    def copy_to_host(self, ary=None, stream=0):
        """Copy ``self`` to ``ary`` or create a new Numpy ndarray
        if ``ary`` is ``None``.

        If a CUDA ``stream`` is given, then the transfer will be made
        asynchronously as part as the given stream.  Otherwise, the transfer is
        synchronous: the function returns after the copy is finished.

        Always returns the host array.

        Example::

            import numpy as np
            from numba import cuda

            arr = np.arange(1000)
            d_arr = cuda.to_device(arr)

            my_kernel[100, 100](d_arr)

            result_array = d_arr.copy_to_host()
        """
        stream = self._default_stream(stream)
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

        assert self.alloc_size >= 0, "Negative memory size"
        if self.alloc_size != 0:
            _driver.device_to_host(hostary, self, self.alloc_size, stream=stream)

        if ary is None:
            if self.size == 0:
                hostary = np.ndarray(shape=self.shape, dtype=self.dtype,
                                     buffer=hostary)
            else:
                hostary = np.ndarray(shape=self.shape, dtype=self.dtype,
                                     strides=self.strides, buffer=hostary)
        return hostary

    def to_host(self, stream=0):
        stream = self._default_stream(stream)
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
        stream = self._default_stream(stream)
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
        return self.gpu_data

    def get_ipc_handle(self):
        """
        Returns a *IpcArrayHandle* object that is safe to serialize and transfer
        to another process to share the local allocation.

        Note: this feature is only available on Linux.
        """
        ipch = devices.get_context().get_ipc_handle(self.gpu_data)
        desc = dict(shape=self.shape, strides=self.strides, dtype=self.dtype)
        return IpcArrayHandle(ipc_handle=ipch, array_desc=desc)

    def view(self, dtype):
        """Returns a new object by reinterpretting the dtype without making a
        copy of the data.
        """
        dtype = np.dtype(dtype)
        if dtype.itemsize != self.dtype.itemsize:
            raise TypeError("new dtype itemsize doesn't match")
        return DeviceNDArray(
            shape=self.shape,
            strides=self.strides,
            dtype=dtype,
            stream=self.stream,
            gpu_data=self.gpu_data,
            )


class DeviceRecord(DeviceNDArrayBase):
    '''
    An on-GPU record type
    '''
    def __init__(self, dtype, stream=0, gpu_data=None):
        shape = ()
        strides = ()
        super(DeviceRecord, self).__init__(shape, strides, dtype, stream,
                                           gpu_data)

    @property
    def flags(self):
        """
        For `numpy.ndarray` compatibility. Ideally this would return a
        `np.core.multiarray.flagsobj`, but that needs to be constructed
        with an existing `numpy.ndarray` (as the C- and F- contiguous flags
        aren't writeable).
        """
        return dict(self._dummy.flags) # defensive copy

    @property
    def _numba_type_(self):
        """
        Magic attribute expected by Numba to get the numba type that
        represents this object.
        """
        return numpy_support.from_dtype(self.dtype)


@lru_cache
def _assign_kernel(ndim):
    """
    A separate method so we don't need to compile code every assignment (!).

    :param ndim: We need to have static array sizes for cuda.local.array, so
        bake in the number of dimensions into the kernel
    """
    from numba import cuda  # circular!

    @cuda.jit
    def kernel(lhs, rhs):
        location = cuda.grid(1)

        n_elements = 1
        for i in range(lhs.ndim):
            n_elements *= lhs.shape[i]
        if location >= n_elements:
            # bake n_elements into the kernel, better than passing it in
            # as another argument.
            return

        # [0, :] is the to-index (into `lhs`)
        # [1, :] is the from-index (into `rhs`)
        idx = cuda.local.array(
            shape=(2, ndim),
            dtype=types.int64)

        for i in range(ndim - 1, -1, -1):
            idx[0, i] = location % lhs.shape[i]
            idx[1, i] = (location % lhs.shape[i]) * (rhs.shape[i] > 1)
            location //= lhs.shape[i]

        lhs[to_fixed_tuple(idx[0], ndim)] = rhs[to_fixed_tuple(idx[1], ndim)]
    return kernel


class DeviceNDArray(DeviceNDArrayBase):
    '''
    An on-GPU array type
    '''
    def is_f_contiguous(self):
        '''
        Return true if the array is Fortran-contiguous.
        '''
        return self._dummy.is_f_contig

    @property
    def flags(self):
        """
        For `numpy.ndarray` compatibility. Ideally this would return a
        `np.core.multiarray.flagsobj`, but that needs to be constructed
        with an existing `numpy.ndarray` (as the C- and F- contiguous flags
        aren't writeable).
        """
        return dict(self._dummy.flags) # defensive copy

    def is_c_contiguous(self):
        '''
        Return true if the array is C-contiguous.
        '''
        return self._dummy.is_c_contig

    def __array__(self, dtype=None):
        """
        :return: an `numpy.ndarray`, so copies to the host.
        """
        return self.copy_to_host().__array__(dtype)

    def __len__(self):
        return self.shape[0]

    def reshape(self, *newshape, **kws):
        """
        Reshape the array without changing its contents, similarly to
        :meth:`numpy.ndarray.reshape`. Example::

            d_arr = d_arr.reshape(20, 50, order='F')
        """
        if len(newshape) == 1 and isinstance(newshape[0], (tuple, list)):
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
        '''
        Flatten the array without changing its contents, similar to
        :meth:`numpy.ndarray.ravel`.
        '''
        stream = self._default_stream(stream)
        cls = type(self)
        newarr, extents = self._dummy.ravel(order=order)

        if extents == [self._dummy.extent]:
            return cls(shape=newarr.shape, strides=newarr.strides,
                       dtype=self.dtype, gpu_data=self.gpu_data,
                       stream=stream)

        else:
            raise NotImplementedError("operation requires copying")

    @devices.require_context
    def __getitem__(self, item):
        return self._do_getitem(item)

    def getitem(self, item, stream=0):
        """Do `__getitem__(item)` with CUDA stream
        """
        return self._do_getitem(item, stream)

    def _do_getitem(self, item, stream=0):
        stream = self._default_stream(stream)

        arr = self._dummy.__getitem__(item)
        extents = list(arr.iter_contiguous_extent())
        cls = type(self)
        if len(extents) == 1:
            newdata = self.gpu_data.view(*extents[0])

            if not arr.is_array:
                # Element indexing
                hostary = np.empty(1, dtype=self.dtype)
                _driver.device_to_host(dst=hostary, src=newdata,
                                       size=self._dummy.itemsize,
                                       stream=stream)
                return hostary[0]
            else:
                return cls(shape=arr.shape, strides=arr.strides,
                           dtype=self.dtype, gpu_data=newdata, stream=stream)
        else:
            newdata = self.gpu_data.view(*arr.extent)
            return cls(shape=arr.shape, strides=arr.strides,
                       dtype=self.dtype, gpu_data=newdata, stream=stream)

    @devices.require_context
    def __setitem__(self, key, value):
        return self._do_setitem(key, value)

    def setitem(self, key, value, stream=0):
        """Do `__setitem__(key, value)` with CUDA stream
        """
        return self._so_getitem(key, value, stream)

    def _do_setitem(self, key, value, stream=0):

        stream = self._default_stream(stream)

        # (1) prepare LHS

        arr = self._dummy.__getitem__(key)
        newdata = self.gpu_data.view(*arr.extent)

        if isinstance(arr, dummyarray.Element):
            # convert to a 1d array
            shape = (1,)
            strides = (self.dtype.itemsize,)
        else:
            shape = arr.shape
            strides = arr.strides

        lhs = type(self)(
            shape=shape,
            strides=strides,
            dtype=self.dtype,
            gpu_data=newdata,
            stream=stream)

        # (2) prepare RHS

        rhs, _ = auto_device(value, stream=stream)
        if rhs.ndim > lhs.ndim:
            raise ValueError("Can't assign %s-D array to %s-D self" % (
                rhs.ndim,
                lhs.ndim))
        rhs_shape = np.ones(lhs.ndim, dtype=np.int64)
        rhs_shape[-rhs.ndim:] = rhs.shape
        rhs = rhs.reshape(*rhs_shape)
        for i, (l, r) in enumerate(zip(lhs.shape, rhs.shape)):
            if r != 1 and l != r:
                raise ValueError("Can't copy sequence with size %d to array axis %d with dimension %d" % (
                    r,
                    i,
                    l))

        # (3) do the copy

        n_elements = np.prod(lhs.shape)
        _assign_kernel(lhs.ndim).forall(n_elements, stream=stream)(lhs, rhs)


class IpcArrayHandle(object):
    """
    An IPC array handle that can be serialized and transfer to another process
    in the same machine for share a GPU allocation.

    On the destination process, use the *.open()* method to creates a new
    *DeviceNDArray* object that shares the allocation from the original process.
    To release the resources, call the *.close()* method.  After that, the
    destination can no longer use the shared array object.  (Note: the
    underlying weakref to the resource is now dead.)

    This object implements the context-manager interface that calls the
    *.open()* and *.close()* method automatically::

        with the_ipc_array_handle as ipc_array:
            # use ipc_array here as a normal gpu array object
            some_code(ipc_array)
        # ipc_array is dead at this point
    """
    def __init__(self, ipc_handle, array_desc):
        self._array_desc = array_desc
        self._ipc_handle = ipc_handle

    def open(self):
        """
        Returns a new *DeviceNDArray* that shares the allocation from the
        original process.  Must not be used on the original process.
        """
        dptr = self._ipc_handle.open(devices.get_context())
        return DeviceNDArray(gpu_data=dptr, **self._array_desc)

    def close(self):
        """
        Closes the IPC handle to the array.
        """
        self._ipc_handle.close()

    def __enter__(self):
        return self.open()

    def __exit__(self, type, value, traceback):
        self.close()


class MappedNDArray(DeviceNDArrayBase, np.ndarray):
    """
    A host array that uses CUDA mapped memory.
    """

    def device_setup(self, gpu_data, stream=0):
        self.gpu_data = gpu_data


def from_array_like(ary, stream=0, gpu_data=None):
    "Create a DeviceNDArray object that is like ary."
    if ary.ndim == 0:
        ary = ary.reshape(1)
    return DeviceNDArray(ary.shape, ary.strides, ary.dtype,
                         writeback=ary, stream=stream, gpu_data=gpu_data)


def from_record_like(rec, stream=0, gpu_data=None):
    "Create a DeviceRecord object that is like rec."
    return DeviceRecord(rec.dtype, stream=stream, gpu_data=gpu_data)


errmsg_contiguous_buffer = ("Array contains non-contiguous buffer and cannot "
                            "be transferred as a single memory region. Please "
                            "ensure contiguous buffer with numpy "
                            ".ascontiguousarray()")


def sentry_contiguous(ary):
    if not ary.flags['C_CONTIGUOUS'] and not ary.flags['F_CONTIGUOUS']:
        if ary.strides[0] == 0:
            # Broadcasted, ensure inner contiguous
            return sentry_contiguous(ary[0])

        else:
            raise ValueError(errmsg_contiguous_buffer)


def auto_device(obj, stream=0, copy=True):
    """
    Create a DeviceRecord or DeviceArray like obj and optionally copy data from
    host to device. If obj already represents device memory, it is returned and
    no copy is made.
    """
    if _driver.is_device_memory(obj):
        return obj, False
    elif hasattr(obj, '__cuda_array_interface__'):
        return numba.cuda.as_cuda_array(obj), False
    else:
        if isinstance(obj, np.void):
            devobj = from_record_like(obj, stream=stream)
        else:
            # This allows you to pass non-array objects like constants
            # and objects implementing the
            # [array interface](https://docs.scipy.org/doc/numpy-1.13.0/reference/arrays.interface.html)
            # into this function (with no overhead -- copies -- for `obj`s
            # that are already `ndarray`s.
            obj = np.array(
                obj,
                copy=False,
                subok=True)
            sentry_contiguous(obj)
            devobj = from_array_like(obj, stream=stream)
        if copy:
            devobj.copy_to_device(obj, stream=stream)
        return devobj, True

