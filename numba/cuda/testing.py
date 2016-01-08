from __future__ import print_function, absolute_import, division

import contextlib
import io
import os
import sys

from numba import config, unittest_support as unittest
from numba.tests.support import captured_stdout


class CUDATestCase(unittest.TestCase):
    def tearDown(self):
        from numba.cuda.cudadrv.devices import reset

        reset()


def skip_on_cudasim(reason):
    return unittest.skipIf(config.ENABLE_CUDASIM, reason)


@contextlib.contextmanager
def redirect_fd(fd):
    """
    Temporarily redirect *fd* to a pipe's write end and return a file object
    wrapping the pipe's read end.
    """
    save = os.dup(fd)
    r, w = os.pipe()
    try:
        os.dup2(w, fd)
        yield io.open(r, "r")
    finally:
        os.close(w)
        os.dup2(save, fd)
        os.close(save)


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
        fd = sys.__stdout__.fileno()
        with redirect_fd(fd) as stream:
            yield CUDATextCapture(stream)
            cuda.synchronize()
