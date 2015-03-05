from __future__ import print_function, division, absolute_import

import atexit
import timeit
import functools
import tempfile
from contextlib import contextmanager


class Profiler(object):
    """
    Timestamp based profiler
    """

    def __init__(self, timer=None):
        self._timer = timeit.default_timer if timer is None else timer
        self._buf = tempfile.NamedTemporaryFile(mode='w', suffix='.profile',
                                                prefix='numba.', delete=False)
        self._first_timestamp = self._timer()
        # Register atexit handling
        atexit.register(self.exit)

    def _timestamp(self):
        return self._timer() - self._first_timestamp

    def exit(self):
        print("\n\nprofile data written to", self._buf.name)
        self._buf.close()

    def start(self, evt):
        print('S', self._timestamp(), evt, file=self._buf)

    def end(self, evt):
        print('E', self._timestamp(), evt, file=self._buf)

    @contextmanager
    def record(self, evt):
        self.start(evt)
        yield
        self.end(evt)

    def mark(self, evt):
        def decor(func):
            @functools.wraps(func)
            def wrapped(*args, **kwargs):
                with self.record(evt):
                    return func(*args, **kwargs)

            return wrapped

        return decor


class DummyProfiler(object):
    def __init__(self, *args, **kwargs):
        pass

    def start(self, *args, **kwargs):
        pass

    def end(self, *args, **kwargs):
        pass

    @contextmanager
    def record(self, *args, **kargs):
        yield

    def mark(self, evt):
        def decor(func):
            @functools.wraps(func)
            def wrapped(*args, **kwargs):
                return func(*args, **kwargs)

            return wrapped

        return decor
