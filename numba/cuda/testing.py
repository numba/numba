import contextlib
import os
import platform
import shutil
import sys

from numba.tests.support import (
    captured_stdout,
    SerialMixin,
    redirect_c_stdout,
)
from numba.cuda.cuda_paths import get_conda_ctk
from numba.cuda.cudadrv import devices, libs
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

    def setUp(self):
        self._low_occupancy_warnings = config.CUDA_LOW_OCCUPANCY_WARNINGS
        self._warn_on_implicit_copy = config.CUDA_WARN_ON_IMPLICIT_COPY

        # Disable warnings about low gpu utilization in the test suite
        config.CUDA_LOW_OCCUPANCY_WARNINGS = 0
        # Disable warnings about host arrays in the test suite
        config.CUDA_WARN_ON_IMPLICIT_COPY = 0

    def tearDown(self):
        config.CUDA_LOW_OCCUPANCY_WARNINGS = self._low_occupancy_warnings
        config.CUDA_WARN_ON_IMPLICIT_COPY = self._warn_on_implicit_copy


class ContextResettingTestCase(CUDATestCase):
    """
    For tests where the context needs to be reset after each test. Typically
    these inspect or modify parts of the context that would usually be expected
    to be internal implementation details (such as the state of allocations and
    deallocations, etc.).
    """

    def tearDown(self):
        super().tearDown()
        from numba.cuda.cudadrv.devices import reset
        reset()


def ensure_supported_ccs_initialized():
    from numba.cuda import is_available as cuda_is_available
    from numba.cuda.cudadrv import nvvm

    if cuda_is_available():
        # Ensure that cudart.so is loaded and the list of supported compute
        # capabilities in the nvvm module is populated before a fork. This is
        # needed because some compilation tests don't require a CUDA context,
        # but do use NVVM, and it is required that libcudart.so should be
        # loaded before a fork (note that the requirement is not explicitly
        # documented).
        nvvm.get_supported_ccs()


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


def skip_without_nvdisasm(reason):
    nvdisasm_path = shutil.which('nvdisasm')
    return unittest.skipIf(nvdisasm_path is None, reason)


def skip_with_nvdisasm(reason):
    nvdisasm_path = shutil.which('nvdisasm')
    return unittest.skipIf(nvdisasm_path is not None, reason)


def skip_on_arm(reason):
    cpu = platform.processor()
    is_arm = cpu.startswith('arm') or cpu.startswith('aarch')
    return unittest.skipIf(is_arm, reason)


def cc_X_or_above(major, minor):
    if not config.ENABLE_CUDASIM:
        cc = devices.get_context().device.compute_capability
        return cc >= (major, minor)
    else:
        return True


def skip_unless_cc_32(fn):
    return unittest.skipUnless(cc_X_or_above(3, 2), "requires cc >= 3.2")(fn)


def skip_unless_cc_50(fn):
    return unittest.skipUnless(cc_X_or_above(5, 0), "requires cc >= 5.0")(fn)


def skip_unless_cc_60(fn):
    return unittest.skipUnless(cc_X_or_above(6, 0), "requires cc >= 6.0")(fn)


def cudadevrt_missing():
    if config.ENABLE_CUDASIM:
        return False
    try:
        libs.check_static_lib('cudadevrt')
    except FileNotFoundError:
        return True
    return False


def skip_if_cudadevrt_missing(fn):
    return unittest.skipIf(cudadevrt_missing(), 'cudadevrt missing')(fn)


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

    def __init__(self, arr):
        self._arr = arr
        self.__cuda_array_interface__ = arr.__cuda_array_interface__
