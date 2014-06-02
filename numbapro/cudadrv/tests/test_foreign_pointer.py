from __future__ import print_function, absolute_import
import numpy
from numbapro import cuda
from numba.cuda.cudadrv.devicearray import DeviceNDArray
from numbapro.testsupport import unittest

# Any object that defines the following attributes can be used as a CUDA memory
# buffer object in NumbaPro:
#
#  Attributes:
#       __cuda_memory__ : Must be True
#       device_ctypes_pointer : a ctypes.c_void_p object containing a pointer
#                               to a CUDA memory buffer
#

class ForeignCTypePointer(object):
    '''This class allows "borrowing" a CUDA memory pointer to be used in
    NumbaPro.
    '''
    __cuda_memory__ = True      # necessary for the interface

    def __init__(self, ptr):
        self.ptr = ptr

    @property
    def device_ctypes_pointer(self):   # necessary for the interface
        return self.ptr


class TestForeignPointer(unittest.TestCase):
    def test_foreign_pointer(self):
        # We will steal a ctypes pointer from a device array for demonstration
        # purpose.
        # You can replace `ptr` with any ctypes.c_void_p object that points
        # to an actual CUDA memory buffer created by the CUDA driver or CUDA runtime
        # API.
        dary = cuda.device_array(shape=10, dtype=numpy.float32)
        print(dary.device_ctypes_pointer)
        ptr = dary.device_ctypes_pointer

        # At this point, `ptr` has a valid CUDA pointer.
        # We can create a DeviceNDArray object directly given the shape, strides,
        # numpy.dtype and `ptr`.
        shape = 10,                         # array shape
        strides = 4,                        # array strides
        dtype = numpy.dtype('float32')      # array element dtype
        foreignptr = ForeignCTypePointer(ptr=ptr)

        # allocate a device ndarray to be used in numbapro
        custom_ary = DeviceNDArray(shape=shape, strides=strides, dtype=dtype,
                                   gpu_data=foreignptr)

        print(custom_ary.copy_to_host())

        # More notes:
        # DeviceNDArray does not deallocate the CUDA pointer.
        # The object for `gpu_data` argument can bind the deallocation of the CUDA
        # pointer to its lifetime.


if __name__ == '__main__':
    unittest.main()
