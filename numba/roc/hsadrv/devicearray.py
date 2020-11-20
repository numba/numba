"""
A HSA dGPU backed ND Array is recognized by checking the __hsa_memory__
attribute on the object.  If it exists and evaluate to True, it must define
shape, strides, dtype and size attributes similar to a NumPy ndarray.
"""
import warnings
import math
import copy
import weakref
from ctypes import c_void_p
import numpy as np
from numba.roc.hsadrv import driver as _driver
from numba.roc.hsadrv import devices
from numba.core import types
from .error import HsaContextMismatchError
from numba.misc import dummyarray
from numba.np import numpy_support


def is_hsa_ndarray(obj):
    "Check if an object is a HSA ndarray"
    return getattr(obj, '__hsa_ndarray__', False)


def verify_hsa_ndarray_interface(obj):
    "Verify the HSA ndarray interface for an obj"
    require_hsa_ndarray(obj)

    def requires_attr(attr, typ):
        if not hasattr(obj, attr):
            raise AttributeError(attr)
        if not isinstance(getattr(obj, attr), typ):
            raise AttributeError('%s must be of type %s' % (attr, typ))

    requires_attr('shape', tuple)
    requires_attr('strides', tuple)
    requires_attr('dtype', np.dtype)
    requires_attr('size', int)


def require_hsa_ndarray(obj):
    "Raises ValueError if is_hsa_ndarray(obj) evaluates False"
    if not is_hsa_ndarray(obj):
        raise ValueError('require an hsa ndarray object')


class DeviceNDArrayBase(object):
    """Base class for an on dGPU NDArray representation cf. numpy.ndarray
    """
    __hsa_memory__ = True
    __hsa_ndarray__ = True     # There must be dgpu_data attribute as a result

    def __init__(self, shape, strides, dtype, dgpu_data=None):
        """
        Args
        ----

        shape
            array shape.
        strides
            array strides.
        dtype
            data type as numpy.dtype.
        dgpu_data
            user provided device memory for the ndarray data buffer
        """
        if isinstance(shape, int):
            shape = (shape,)
        if isinstance(strides, int):
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
        # prepare dgpu memory
        if self.size > 0:
            if dgpu_data is None:
                from numba.roc.api import _memory_size_from_info
                self.alloc_size = _memory_size_from_info(self.shape,
                                          self.strides, self.dtype.itemsize)
                # find a coarse region on the dGPU
                dgpu_data = devices.get_context().mempoolalloc(self.alloc_size)
            else:  # we have some preallocated dgpu_memory
                sz = getattr(dgpu_data, '_hsa_memsize_', None)
                if sz is None:
                    raise ValueError('dgpu_data as no _hsa_memsize_ attribute')
                assert sz >= 0
                self.alloc_size = sz
        else:
            dgpu_data = None
            self.alloc_size = 0

        self.dgpu_data = dgpu_data

    @property
    def _context(self):
        return self.dgpu_data.context

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
        if self.dgpu_data is None:
            return c_void_p(0)
        else:
            return self.dgpu_data.device_ctypes_pointer

    def copy_to_device(self, ary, stream=None, context=None):
        """Copy `ary` to `self`.

        If `ary` is a HSA memory, perform a device-to-device transfer.
        Otherwise, perform a a host-to-device transfer.

        If `stream` is a stream object, an async copy to used.
        """
        if ary.size == 0:
            # Nothing to do
            return

        if context is not None:
            if self.dgpu_data is not None:
                expect, got = self._context, context
                if expect.unproxy != got.unproxy:
                    raise HsaContextMismatchError(expect=expect, got=got)
        else:
            context = self._context

        # TODO: Worry about multiple dGPUs
        #if _driver.is_device_memory(ary):
        #    sz = min(self.alloc_size, ary.alloc_size)
        #    _driver.device_to_device(self, ary, sz)
        #else:
        #    sz = min(_driver.host_memory_size(ary), self.alloc_size)

        sz = self.alloc_size

        # host_to_dGPU(context, dst, src, size):
        if stream is None:
            _driver.hsa.implicit_sync()

            if isinstance(ary, DeviceNDArray):
                _driver.dGPU_to_dGPU(self._context, self, ary, sz)
            else:
                _driver.host_to_dGPU(self._context, self, ary, sz)
        else:
            if isinstance(ary, DeviceNDArray):
                _driver.async_dGPU_to_dGPU(dst_ctx=self._context,
                                           src_ctx=ary._context,
                                           dst=self, src=ary, size=sz,
                                           stream=stream)
            else:
                _driver.async_host_to_dGPU(dst_ctx=self._context,
                                        src_ctx=devices.get_cpu_context(),
                                        dst=self, src=ary, size=sz,
                                        stream=stream)

    def copy_to_host(self, ary=None, stream=None):
        """Copy ``self`` to ``ary`` or create a new Numpy ndarray
        if ``ary`` is ``None``.

        The transfer is synchronous: the function returns after the copy
        is finished.

        Always returns the host array.

        Example::

            import numpy as np
            from numba import hsa

            arr = np.arange(1000)
            d_arr = hsa.to_device(arr)

            my_kernel[100, 100](d_arr)

            result_array = d_arr.copy_to_host()
        """
        if ary is None:  # destination does not exist
            hostary = np.empty(shape=self.alloc_size, dtype=np.byte)
        else: # destination does exist, it's `ary`, check it
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
            hostary = ary  # this is supposed to be a ptr for writing

        # a location for the data exists as `hostary`
        assert self.alloc_size >= 0, "Negative memory size"

        context = self._context

        # copy the data from the device to the hostary
        if self.alloc_size != 0:
            sz = self.alloc_size
            if stream is None:
                _driver.hsa.implicit_sync()
                _driver.dGPU_to_host(context, hostary, self, sz)
            else:
                _driver.async_dGPU_to_host(dst_ctx=devices.get_cpu_context(),
                                           src_ctx=self._context,
                                           dst=hostary, src=self,
                                           size=sz, stream=stream)

        # if the location for the data was originally None
        # then create a new ndarray and plumb in the new memory
        if ary is None:
            if self.size == 0:
                hostary = np.ndarray(shape=self.shape, dtype=self.dtype,
                                     buffer=hostary)
            else:
                hostary = np.ndarray(shape=self.shape, dtype=self.dtype,
                                     strides=self.strides, buffer=hostary)
        else: # else hostary points to ary and how has the right memory
            hostary = ary

        return hostary

    def as_hsa_arg(self):
        """Returns a device memory object that is used as the argument.
        """
        return self.dgpu_data


class DeviceNDArray(DeviceNDArrayBase):
    '''
    An on-dGPU array type
    '''
    def is_f_contiguous(self):
        '''
        Return true if the array is Fortran-contiguous.
        '''
        return self._dummy.is_f_contig

    def is_c_contiguous(self):
        '''
        Return true if the array is C-contiguous.
        '''
        return self._dummy.is_c_contig

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
                       dtype=self.dtype, dgpu_data=self.dgpu_data)

        newarr, extents = self._dummy.reshape(*newshape, **kws)

        if extents == [self._dummy.extent]:
            return cls(shape=newarr.shape, strides=newarr.strides,
                       dtype=self.dtype, dgpu_data=self.dgpu_data)
        else:
            raise NotImplementedError("operation requires copying")

    def ravel(self, order='C'):
        '''
        Flatten the array without changing its contents, similar to
        :meth:`numpy.ndarray.ravel`.
        '''
        cls = type(self)
        newarr, extents = self._dummy.ravel(order=order)

        if extents == [self._dummy.extent]:
            return cls(shape=newarr.shape, strides=newarr.strides,
                       dtype=self.dtype, dgpu_data=self.dgpu_data)

        else:
            raise NotImplementedError("operation requires copying")


class HostArray(np.ndarray):
    __hsa_memory__ = True

    @property
    def device_ctypes_pointer(self):
        return self.ctypes.data_as(c_void_p)


def from_array_like(ary, dgpu_data=None):
    "Create a DeviceNDArray object that is like ary."
    if ary.ndim == 0:
        ary = ary.reshape(1)
    return DeviceNDArray(ary.shape, ary.strides, ary.dtype,
                         dgpu_data=dgpu_data)



errmsg_contiguous_buffer = ("Array contains non-contiguous buffer and cannot "
                            "be transferred as a single memory region. Please "
                            "ensure contiguous buffer with numpy "
                            ".ascontiguousarray()")


def _single_buffer(ary):
    i = np.argmax(ary.strides)
    size = ary.strides[i] * ary.shape[i]
    return size == ary.nbytes


def sentry_contiguous(ary):
    if not ary.flags['C_CONTIGUOUS'] and not ary.flags['F_CONTIGUOUS']:
        if ary.strides[0] == 0:
            # Broadcasted, ensure inner contiguous
            return sentry_contiguous(ary[0])

        elif _single_buffer(ary):
            return True

        else:
            raise ValueError(errmsg_contiguous_buffer)


def auto_device(obj, context, stream=None, copy=True):
    """
    Create a DeviceArray like obj and optionally copy data from
    host to device. If obj already represents device memory, it is returned and
    no copy is made.
    """
    if _driver.is_device_memory(obj): # it's already on the dGPU
        return obj, False
    else: # needs to be copied to the dGPU
        sentry_contiguous(obj)
        devobj = from_array_like(obj)
        if copy:
            devobj.copy_to_device(obj, stream=stream, context=context)
        return devobj, True


