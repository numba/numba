import unittest

from numba import njit
from numba.core.event import install_recorder, RecordingListener
from numba.tests.support import TestCase


class TestEvent(TestCase):
    def test_recording_listener(self):
        @njit
        def foo(x):
            return x + x

        with install_recorder("numba:compile") as rec:
            foo(1)

        self.assertIsInstance(rec, RecordingListener)
        # Check there must be at least two event.
        # Because there must be a START and END for the compilation of foo()
        self.assertGreaterEqual(len(rec.buffer), 2)

    def test_compiler_lock_event(self):
        @njit
        def foo(x):
            return x + x

        foo(1)
        md = foo.get_metadata(foo.signatures[0])
        lock_duration = md['timers']['compiler_lock']
        self.assertIsInstance(lock_duration, float)
        self.assertGreater(lock_duration, 0)


if __name__ == "__main__":
    unittest.main()
