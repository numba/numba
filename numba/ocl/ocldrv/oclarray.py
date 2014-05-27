"""
An OpenCL N-dimensional array. This is similar (and based on) the CUDA
ND Array defined in the cuda driver.

OpenCL minimizes implicit state, and so does the driver we use. That
means that some stuff that is implicit in the CUDA version becomes
explicit around here. Most notably, the context.

An OpenCL ND Array is recognized by checking the __ocl_memory__ on
the object class. A valid OpenCL ND Array defines shape, strides,
dtype and size in the same way that a NumPy ndarray would.
"""

from __future__ import print_function, absolute_import, division

from . import MemObject
from .types import *
from ... import dummyarray
import numpy as np
from . import oclmem

try:
    long
except NameError:
    long = int


def is_ocl_ndarray(obj):
    "Check if an object is an OpenCL ndarray"
    return getattr(obj, '__ocl_ndarray__', False)

def require_ocl_ndarray(obj):
    "Raises ValueError is is_cuda_ndarray(obj) evaluates False"
    if not is_ocl_ndarray(obj):
        raise ValueError('require an OpenCL ndarray object')

def verify_ocl_ndarray_interface(obj):
    "Verify the OpenCL ndarray interface for an obj"
    require_ocl_ndarray(obj)

    def requires_attr(attr, typ):
        if not hasattr(obj, attr):
            raise AttributeError(attr)
        if not isinstance(getattr(obj, attr), typ):
            raise AttributeError('%s must be of type %s' % (attr, typ))

    requires_attr('shape', tuple)
    requires_attr('strides', tuple)
    requires_attr('dtype', np.dtype)


def _array_desc_ctype(ndims):
    class c_array(ctypes.Structure):
        # cl_long (64 bit) used, better be generous
        _fields_ = [('shape', cl_long * ndims),
                    ('strides', cl_long * ndims)]

    return c_array

def _create_ocl_desc(context, shape, strides):
    """creates an on-context memory buffer for an array descriptor"""
    desc = _array_desc_ctype(len(shape))()
    desc.shape = shape
    desc.strides = strides
    sz = ctypes.sizeof(desc)
    ptr = ctypes.addressof(desc)
    buff = context.create_buffer_and_copy(sz, ptr)
    return buff

class OpenCLNDArrayBase(object):
    """
    An Array accessible to OpenCL compute units.

    An Array consists of two different OpenCL buffers:
    - desc: a descriptor for how the array data is organized (shape, strides...)
    - data: the actual data, arranged as described by the array_desc

    This object __init__ must only be used by the functions that are used as
    constructors, as it is quite low-level.
    """
    __ocl_memory__ = True
    __ocl_ndarray__ = True # There must be a _desc and a _data attribute

    def __init__(self, shape, strides, dtype, ocl_desc, ocl_data):
        """
        Args
        ----
        shape
            array shape.
        strides
            array strides.
        dtype
            data type as numpy.dtype.
        ocl_desc
            user provided device memory for the ndarray head structure
        ocl_data
            user provided device memory for the ndarray data buffer
        """
        assert(isinstance(shape, tuple))
        assert(isinstance(strides, tuple))
        assert(len(strides) == len(shape))
        assert(isinstance(dtype, np.dtype))
        assert(isinstance(ocl_desc, MemObject))
        assert(isinstance(ocl_data, MemObject))

        self.ndim = len(shape)
        self.shape = shape
        self.strides = strides
        self.dtype = dtype
        self._desc = ocl_desc
        self._data = ocl_data
        self._dummy = dummyarray.Array.from_desc(0, shape, strides,
                                                 dtype.itemsize)

    @property
    def alloc_size(self):
        return self.data.size

    @property
    def _memobject_(self):
        """return the main memobject associated to the main memory"""
        return self._data

    @property
    def data(self):
        return self._data

    @property
    def desc(self):
        return self._desc

    @property
    def ocl_arg(self):
        return self._desc.id, self._data.id


    def copy_to_device(self, ary, queue=None):
        """Copy `ary` to `self`.

        If `ary` is an in-device array, perform a device-to-device transfer.
        Otherwise, perform a a host-to-device transfer.
        """
        if queue is None:
            q = self._data.context.create_command_queue(self._data.context.devices[0])

        if oclmem.is_device_memory(ary):
            sz = min(self._memobject_.size, ary._memobject_.size)
            q.enqueue_copy_buffer(ary._memobject_, self._memobject_, 0, 0, sz)
        else:
            sz = min(oclmem.host_memory_size(ary), self._data.size)
            q.enqueue_write_buffer(self._data, 0, sz, oclmem.host_pointer(ary))


    def copy_to_host(self, ary=None, queue=None):
        """Copy ``self`` to ``ary`` or create a new numpy ndarray
        if ``ary`` is ``None``.

        Always returns the host array.
        """
        if ary is None:
            hostary = np.empty(shape=self._data.size, dtype=np.byte)
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

            if ary.nbytes != self._data.size:
                raise TypeError('buffer sizes do not match: device: {0} host: {1}'.format(self._data.size, ary.nbytes))
            hostary = ary

        if queue is None:
            ctxt = self._data.context
            queue = ctxt.create_command_queue(ctxt.devices[0])

        queue.enqueue_read_buffer(self._data, 0, self._data.size, hostary.ctypes.data)

        if ary is None:
            hostary = np.ndarray(shape=self.shape, strides=self.strides,
                                 dtype=self.dtype, buffer=hostary)
        return hostary

    def split(self, section, stream=0):
        """Split the array into equal partition of the `section` size.
        If the array cannot be equally divided, the last section will be
        smaller.
        """
        if self.ndim != 1:
            raise ValueError("only support 1d array")
        if self.strides[0] != self.dtype.itemsize:
            raise ValueError("only support unit stride")

        ctxt = self._data.context
        strides = self.strides
        itemsize = self.dtype.itemsize
        total = self.shape[0]
        for begin in range(0, total, section):
            end = min(begin + section, total)
            shape = (end - begin,)
            cl_desc = _create_ocl_desc(ctxt, shape, strides)
            cl_data = self._data.create_region(begin * itemsize, (end-begin) * itemsize)
            yield OpenCLNDArray(shape, strides, self.dtype, cl_desc, cl_data)


class OpenCLNDArray(OpenCLNDArrayBase):
    def is_f_contiguous(self):
        return self._dummy.is_f_contig

    def is_c_contiguous(self):
        return self._dummy.is_c_contig

    def reshape(self, *newshape, **kws):
        """reshape(self, *newshape, order='C'):

        Reshape the array and keeping the original data
        """
        newarr, extents = self._dummy.reshape(*newshape, **kws)
        cls = type(self)

        if extents == [self._dummy.extent]:
            ctxt = self._desc.context
            cl_desc = _create_ocl_desc(ctxt, newarr.shape, newarr.strides)
            return cls(newarr.shape, newarr.strides, self.dtype, cl_desc, self._data)
        else:
            raise NotImplementedError("operation requires copying")


    def ravel(self, order='C', stream=0):
        cls = type(self)
        newarr, extents = self._dummy.ravel(order=order)

        # do not destructively reuse metadata as that will modify the
        # original array

        if extents == [self._dummy.extent]:
            ctxt = self._desc.context
            cl_desc = _create_ocl_desc(ctxt, newarr.shape, newarr.strides)
            return cls(newarr.shape, newarr.strides, self.dtype,
                       cl_desc, self._data)
        else:
            raise NotImplementedError("operation requires copying")

    def __getitem__(self, item):
        arr = self._dummy.__getitem__(item)
        extents = list(arr.iter_contiguous_extent())
        cls = type(self)
        # TODO: in opencl it is not possible to create a region from another
        #       region. There needs to be a way to handle that a getitem from
        #       a submatrix returned by getitem
        if len(extents) == 1:
            # only one dimension...
            extents = extents[0]
            if dummyarray.is_element_indexing(item, self.ndim):
                hostary = np.empty(1, dtype=self.dtype)
                ctxt = self._data.context
                q = ctxt.create_command_queue(ctxt.devices[0])
                q.enqueue_read_buffer(self._data, extents[0], hostary.nbytes,
                                      hostary.ctypes.data)
                return hostary[0]
            else:
                new_desc = _create_ocl_desc(self._data.context, arr.shape, arr.strides)
                new_data = self._data.create_region(extents[0], extents[1]-extents[0])
                return cls(arr.shape, arr.strides, self.dtype, new_desc, new_data)
        else:
            new_data = self._data.create_region(arr.extent.begin, arr.extent.end - arr.extent.begin)
            new_desc = _create_ocl_desc(self._data.context, arr.shape, arr.strides)
            return cls(arr.shape, arr.strides, self.dtype, new_desc, new_data)


class MappedNDArray(OpenCLNDArrayBase, np.ndarray):
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

# OpenCL array constructors ############################################

def from_array(ary, ctxt):
    """
    Create an OpenCLArray object based on Numpy's array 'ary'
    """
    if ary.ndim == 0:
        ary = ary.reshape(1)

    cl_desc = _create_ocl_desc(ctxt, ary.shape, ary.strides)
    cl_data = ctxt.create_buffer_and_copy(ary.nbytes, ary.ctypes.data)
    return OpenCLNDArray(ary.shape, ary.strides, ary.dtype, cl_desc, cl_data)

def from_array_like(ary, ctxt):
    "Create a DeviceNDArray object that is like ary."
    if ary.ndim == 0:
        ary = ary.reshape(1)

    cl_desc = _create_ocl_desc(ctxt, ary.shape, ary.strides)
    cl_data = ctxt.create_buffer(ary.nbytes)
    return OpenCLNDArray(ary.shape, ary.strides, ary.dtype, cl_desc, cl_data)

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
