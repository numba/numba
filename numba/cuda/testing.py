from __future__ import print_function, absolute_import, division

import contextlib
import sys

from numba import config, unittest_support as unittest
from numba.tests.support import (
    captured_stdout,
    SerialMixin,
    redirect_c_stdout,
)


class CUDATestCase(SerialMixin, unittest.TestCase):
    def tearDown(self):
        from numba.cuda.cudadrv.devices import reset

        reset()


def skip_on_cudasim(reason):
    return unittest.skipIf(config.ENABLE_CUDASIM, reason)


def skip_unless_cudasim(reason):
    return unittest.skipUnless(config.ENABLE_CUDASIM, reason)


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
