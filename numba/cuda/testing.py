import contextlib
import os
import sys
import enum

from numba.tests.support import (
    captured_stdout,
    SerialMixin,
    redirect_c_stdout,
)
from numba.cuda.cuda_paths import get_conda_ctk
from numba.core import config
from numba.tests.support import TestCase
import unittest


class CUDATestCase(SerialMixin, TestCase):
    """
    For tests that use a CUDA device. Test methods in a CUDATestCase must not
    be run out of module order, because the ContextResettingTestCase may reset
    the context and destroy resources used by a normal CUDATestCase if any of
    its tests are run between tests from a CUDATestCase.
    """


class ContextResettingTestCase(CUDATestCase):
    """
    For tests where the context needs to be reset after each test. Typically
    these inspect or modify parts of the context that would usually be expected
    to be internal implementation details (such as the state of allocations and
    deallocations, etc.).
    """

    def tearDown(self):
        from numba.cuda.cudadrv.devices import reset
        reset()


def skip_on_cudasim(reason):
    """Skip this test if running on the CUDA simulator"""
    return unittest.skipIf(config.ENABLE_CUDASIM, reason)


def skip_unless_cudasim(reason):
    """Skip this test if running on CUDA hardware"""
    return unittest.skipUnless(config.ENABLE_CUDASIM, reason)


def skip_unless_conda_cudatoolkit(reason):
    """Skip test if the CUDA toolkit was not installed by Conda"""
    return unittest.skipUnless(get_conda_ctk() is not None, reason)


def skip_if_external_memmgr(reason):
    """Skip test if an EMM Plugin is in use"""
    return unittest.skipIf(config.CUDA_MEMORY_MANAGER != 'default', reason)


def skip_under_cuda_memcheck(reason):
    return unittest.skipIf(os.environ.get('CUDA_MEMCHECK') is not None, reason)


class CUDATextCapture(object):

    def __init__(self, stream):
        self._stream = stream

    def getvalue(self):
        return self._stream.read()


class PythonTextCapture(object):

    def __init__(self, stream):
        self._stream = stream

    def getvalue(self):
        return self._stream.getvalue()


@contextlib.contextmanager
def captured_cuda_stdout():
    """
    Return a minimal stream-like object capturing the text output of
    either CUDA or the simulator.
    """
    # Prevent accidentally capturing previously output text
    sys.stdout.flush()

    if config.ENABLE_CUDASIM:
        # The simulator calls print() on Python stdout
        with captured_stdout() as stream:
            yield PythonTextCapture(stream)
    else:
        # The CUDA runtime writes onto the system stdout
        from numba import cuda
        with redirect_c_stdout() as stream:
            yield CUDATextCapture(stream)
            cuda.synchronize()


class ForeignArray(object):
    """
    Class for emulating an array coming from another library through the CUDA
    Array interface. This just hides a DeviceNDArray so that it doesn't look
    like a DeviceNDArray.
    """
    class InterfaceType(enum.Enum):
        Attribute = "Attribute"  #: Provide interface description as attribute
        Callable = "Callable"  #: Provide interface description via callable

    def __init__(self, arr, iftype: InterfaceType = InterfaceType.Attribute):
        self._arr = arr
        desc = arr.__cuda_array_interface__
        desc = desc() if callable(desc) else desc
        self.__cuda_array_interface__ = \
            desc if iftype is ForeignArray.InterfaceType.Attribute \
            else lambda: desc
