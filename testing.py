from __future__ import print_function, absolute_import, division

import contextlib
import sys

from numba import config, unittest_support as unittest
from numba.tests.support import (
    captured_stdout,
    SerialMixin,
    redirect_c_stdout,
)

class DPPYTestCase(SerialMixin, unittest.TestCase):
    def setUp(self):
        #init()
	#TODO
        pass
    def tearDown(self):
        #reset()
	#TODO
        pass

class DPPYTextCapture(object):
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
def captured_dppy_stdout():
    """
    Return a minimal stream-like object capturing the text output of dppy
    """
    # Prevent accidentally capturing previously output text
    sys.stdout.flush()

    from numba import dppy
    with redirect_c_stdout() as stream:
        yield DPPYTextCapture(stream)
