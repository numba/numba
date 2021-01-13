import unittest

from numba import njit, jit
from numba.core import event as ev
from numba.tests.support import TestCase


class TestEvent(TestCase):
    def test_recording_listener(self):
        @njit
        def foo(x):
            return x + x

        with ev.install_recorder("numba:compile") as rec:
            foo(1)

        self.assertIsInstance(rec, ev.RecordingListener)
        # Check there must be at least two events.
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

    def test_llvm_lock_event(self):
        @njit
        def foo(x):
            return x + x

        foo(1)
        md = foo.get_metadata(foo.signatures[0])
        lock_duration = md['timers']['llvm_lock']
        self.assertIsInstance(lock_duration, float)
        self.assertGreater(lock_duration, 0)

    def test_install_listener(self):
        ut = self

        class MyListener(ev.Listener):
            def on_start(self, event):
                ut.assertEqual(event.status, ev.EventStatus.START)
                ut.assertEqual(event.kind, "numba:compile")
                ut.assertIs(event.data["dispatcher"], foo)
                dispatcher = event.data["dispatcher"]
                ut.assertIs(dispatcher, foo)
                # Check that the compiling signature is NOT in the overloads
                ut.assertNotIn(event.data["args"], dispatcher.overloads)

            def on_end(self, event):
                ut.assertEqual(event.status, ev.EventStatus.END)
                ut.assertEqual(event.kind, "numba:compile")
                dispatcher = event.data["dispatcher"]
                ut.assertIs(dispatcher, foo)
                # Check that the compiling signature is in the overloads
                ut.assertIn(event.data["args"], dispatcher.overloads)

        @njit
        def foo(x):
            return x

        listener = MyListener()
        with ev.install_listener("numba:compile", listener) as yielded:
            foo(1)

        # Check that the yielded value is the same listener
        self.assertIs(listener, yielded)

    def test_global_register(self):
        ut = self

        class MyListener(ev.Listener):
            def on_start(self, event):
                ut.assertEqual(event.status, ev.EventStatus.START)
                ut.assertEqual(event.kind, "numba:compile")
                # Check it is the same dispatcher
                dispatcher = event.data["dispatcher"]
                ut.assertIs(dispatcher, foo)
                # Check that the compiling signature is NOT in the overloads
                ut.assertNotIn(event.data["args"], dispatcher.overloads)

            def on_end(self, event):
                ut.assertEqual(event.status, ev.EventStatus.END)
                ut.assertEqual(event.kind, "numba:compile")
                # Check it is the same dispatcher
                dispatcher = event.data["dispatcher"]
                ut.assertIs(dispatcher, foo)
                # Check that the compiling signature is in the overloads
                ut.assertIn(event.data["args"], dispatcher.overloads)

        @njit
        def foo(x):
            return x

        listener = MyListener()
        ev.register("numba:compile", listener)
        foo(1)
        ev.unregister("numba:compile", listener)

    def test_lifted_dispatcher(self):
        @jit
        def foo():
            object()   # to trigger loop-lifting
            c = 0
            for i in range(10):
                c += i
            return c

        with ev.install_recorder("numba:compile") as rec:
            foo()

        # Check that there are 4 events.
        # Two for `foo()` and two for the lifted loop.
        self.assertGreaterEqual(len(rec.buffer), 4)

        cres = foo.overloads[foo.signatures[0]]
        [ldisp] = cres.lifted

        lifted_cres = ldisp.overloads[ldisp.signatures[0]]
        self.assertIsInstance(
            lifted_cres.metadata["timers"]["compiler_lock"],
            float,
        )
        self.assertIsInstance(
            lifted_cres.metadata["timers"]["llvm_lock"],
            float,
        )


if __name__ == "__main__":
    unittest.main()
