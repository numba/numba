#
# Test does not work on some cards.
#
from __future__ import print_function, absolute_import, division
import threading
import numpy as np
from numba import ocl
import unittest

class InOtherThread(object):
    def __init__(self, func, *args, **kwargs):
        self.exc_info = None
        self.result = None
        self.func = func
        self.args = args
        self.kwargs = kwargs
        self.thread = None
        self.thread = threading.Thread(target=self.__entry_func)
        self.thread.start()

    @property
    def __entry_func(self):
        def __entry(*args, **kwargs):
            try:
                self.result = self.func(*self.args, **self.kwargs)
            except Exception as e:
                import sys
                self.exc_info = sys.exc_info()

        return __entry

    @property
    def return_value(self):
        assert self.thread is not None
        self.thread.join()

        if self.exc_info:
            try:
                e = self.exc_info[1].with_traceback(self.exc_info[2])
                exec_string = "raise e"
            except AttributeError:
                # this happens on python2, use three arg raise inside an exec
                # block to prevent python3 complaining about syntax errors
                exec_string = "raise self.exc_info[1], None, self.exc_info[2]"
            exec(exec_string)
        return self.result


class TestSelectDevice(unittest.TestCase):
    @unittest.skip('not yet implemented')
    def test_select_device(self):
        def newthread():
            ocl.select_device(0)
            stream = ocl.stream()

            A = np.arange(100)
            dA = ocl.to_device(A, stream=stream)
            stream.synchronize()
            del dA
            del stream
            assert False
            ocl.close()

        for i in range(10):
            InOtherThread(newthread).return_value


if __name__ == '__main__':
    unittest.main()
