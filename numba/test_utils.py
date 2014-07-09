from __future__ import print_function, absolute_import, division

import threading

class InOtherThread(object):
    """Utility class to execute a function in a different thread

    This is used to test stuff in ocl and cuda drivers that deal
    with thread global state, so that they don't interfere with
    other tests.

    On creation a thread is started that executes the argument
    function with the provided args. The result/exceptio is stored.

    accessing the return_value property will wait for the thread
    to finish, and return whatever was returned by the function.
    If an exception was thrown in the other thread, the exception
    will be re-raised in the main thread using the original
    backtrace.
    """
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
