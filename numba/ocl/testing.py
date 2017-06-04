from __future__ import print_function, absolute_import, division

import contextlib
import io
import os
import sys

from numba import config, unittest_support as unittest
from numba.tests.support import captured_stdout


class OCLTestCase(unittest.TestCase):
    def tearDown(self):
        from numba.ocl.ocldrv.devices import reset

        reset()


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


class OCLTextCapture(object):

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
def captured_ocl_stdout():
    """
    Return a minimal stream-like object capturing the text output of
    either OpenCL or the simulator.
    """
    # Prevent accidentally capturing previously output text
    sys.stdout.flush()

    # The OpenCL runtime writes onto the system stdout
    from numba import ocl
    fd = sys.__stdout__.fileno()
    with redirect_fd(fd) as stream:
        yield OCLTextCapture(stream)
        ocl.synchronize()
