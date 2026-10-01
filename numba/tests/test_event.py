import unittest
import string
import threading

import numpy as np

from numba import njit, jit, literal_unroll
from numba.core import event as ev
from numba.tests.support import TestCase, override_config
from numba.core.utils import _lazy_pformat


class TestEvent(TestCase):

    def setUp(self):
        # Trigger compilation to ensure all listeners are initialized
        njit(lambda: None)()
        self.__registered_listeners = len(ev._registered)

    def tearDown(self):
        # Check there is no lingering listeners
        self.assertEqual(len(ev._registered), self.__registered_listeners)

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

    def test_run_pass_event(self):
        @njit
        def foo(x):
            return x + x

        with ev.install_recorder("numba:run_pass") as recorder:
            foo(2)

        self.assertGreater(len(recorder.buffer), 0)
        for _, event in recorder.buffer:
            # Check that all fields are there
            data = event.data
            self.assertIsInstance(data['name'], str)
            self.assertIsInstance(data['qualname'], str)
            self.assertIsInstance(data['module'], str)
            self.assertIsInstance(data['flags'], _lazy_pformat)
            self.assertIsInstance(data['args'], str)
            self.assertIsInstance(data['return_type'], str)

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
        @jit(forceobj=True)
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

    def test_timing_properties(self):
        a = tuple(string.ascii_lowercase)

        @njit
        def bar(x):
            acc = 0
            for i in literal_unroll(a):
                if i in {'1': x}:
                    acc += 1
                else:
                    acc += np.sqrt(x[0, 0])
            return np.sin(x), acc

        @njit
        def foo(x):
            return bar(np.zeros((x, x)))

        with override_config('LLVM_PASS_TIMINGS', True):
            foo(1)

        def get_timers(fn, prop):
            md = fn.get_metadata(fn.signatures[0])
            return md[prop]

        foo_timers = get_timers(foo, 'timers')
        bar_timers = get_timers(bar, 'timers')
        foo_llvm_timer = get_timers(foo, 'llvm_pass_timings')
        bar_llvm_timer = get_timers(bar, 'llvm_pass_timings')

        # Check: time spent in bar() must be longer than in foo()
        self.assertLess(bar_timers['llvm_lock'],
                        foo_timers['llvm_lock'])
        self.assertLess(bar_timers['compiler_lock'],
                        foo_timers['compiler_lock'])

        # Check: time spent in LLVM itself must be less than in the LLVM lock
        self.assertLess(foo_llvm_timer.get_total_time(),
                        foo_timers['llvm_lock'])
        self.assertLess(bar_llvm_timer.get_total_time(),
                        bar_timers['llvm_lock'])

        # Check: time spent in LLVM lock must be less than in compiler
        self.assertLess(foo_timers['llvm_lock'],
                        foo_timers['compiler_lock'])
        self.assertLess(bar_timers['llvm_lock'],
                        bar_timers['compiler_lock'])

    def test_install_listener_thread_local(self):
        # Listeners installed via install_listener() only receive events
        # broadcast on the thread that entered the context manager.
        kind = "test:thread_local"
        errors = []

        def worker():
            try:
                with ev.install_recorder(kind) as worker_rec:
                    ev.start_event(kind)
                    ev.end_event(kind)
                # The worker's recorder captured both events.
                self.assertEqual(len(worker_rec.buffer), 2)
            except Exception as e:
                errors.append(e)

        with ev.install_recorder(kind) as main_rec:
            t = threading.Thread(target=worker)
            t.start()
            t.join()

        self.assertFalse(errors)
        # The main thread's recorder saw none of the worker's events.
        self.assertEqual(len(main_rec.buffer), 0)

    def test_timing_listener_shared_global_registration(self):
        # Regression test for https://github.com/numba/numba/issues/10564
        # A TimingListener registered process-wide via register() receives
        # events from all threads; it must not crash or corrupt its state
        # when notified concurrently.
        kind = "numba:compiler_lock"
        nthreads = 8
        nrounds = 5000
        barrier = threading.Barrier(nthreads)
        errors = []
        tl = ev.TimingListener()

        def worker():
            try:
                barrier.wait()
                for _ in range(nrounds):
                    ev.start_event(kind)
                    ev.end_event(kind)
            except Exception as e:
                errors.append(e)

        ev.register(kind, tl)
        try:
            threads = [threading.Thread(target=worker)
                       for _ in range(nthreads)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
        finally:
            ev.unregister(kind, tl)

        self.assertFalse(errors)
        # All START events were paired with END events.
        self.assertEqual(tl._depth, 0)

    def test_concurrent_compile_timers(self):
        # Compilations on multiple threads each get correct per-thread,
        # per-compilation lock timings.
        nthreads = 4
        nrounds = 3
        barrier = threading.Barrier(nthreads)
        errors = []

        def worker():
            try:
                barrier.wait()
                for _ in range(nrounds):
                    # Fresh dispatcher => fresh compilation every round.
                    f = njit(lambda x: x * 2 + 1)
                    f(1)
                    md = f.get_metadata(f.signatures[0])
                    compiler_lock = md['timers']['compiler_lock']
                    llvm_lock = md['timers']['llvm_lock']
                    self.assertIsInstance(compiler_lock, float)
                    self.assertIsInstance(llvm_lock, float)
                    self.assertGreater(compiler_lock, 0)
                    self.assertGreater(llvm_lock, 0)
                    self.assertLess(llvm_lock, compiler_lock)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(nthreads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertFalse(errors)

    def test_scoped_timer_isolated_from_other_threads(self):
        # A scoped TimingListener must measure only the events emitted on
        # its own thread. This models the reported bug: a listener gets
        # installed while another thread is mid-event, so the first thing
        # delivered to it is an END whose START it never saw. Pre-fix
        # (global registration), that unpaired END poisons _depth to -1 and
        # the subsequent measurement is never finalized (or is garbage) --
        # deterministically, with or without the GIL. Post-fix, the other
        # thread's events are simply not delivered.
        import time
        from timeit import default_timer as timer

        kind = "test:timing_isolation"
        started = threading.Event()
        installed = threading.Event()
        ended = threading.Event()
        errors = []

        def worker():
            try:
                ev.start_event(kind)
                started.set()
                # Wait until the main thread's listener is installed...
                self.assertTrue(installed.wait(timeout=10))
                # ...then emit the END whose START that listener never saw.
                ev.end_event(kind)
                ended.set()
            except Exception as e:
                errors.append(e)

        t = threading.Thread(target=worker)
        t.start()
        try:
            # Ensure the worker's START was emitted before installing, so
            # that (pre-fix) the installed listener receives an unpaired
            # END -- exactly what happened to timers installed by threads
            # blocked on the compiler lock.
            self.assertTrue(started.wait(timeout=10))
            tl = ev.TimingListener()
            with ev.install_listener(kind, tl):
                installed.set()
                # Ensure the worker's unpaired END was emitted (and,
                # pre-fix, delivered) before this thread's measurement
                # window starts.
                self.assertTrue(ended.wait(timeout=10))
                start = timer()
                ev.start_event(kind)
                time.sleep(0.05)
                ev.end_event(kind)
                elapsed = timer() - start
        finally:
            t.join()

        self.assertFalse(errors)
        # The other thread's events did not disturb pairing.
        self.assertEqual(tl._depth, 0)
        # The measurement was made and covers exactly this thread's window.
        self.assertTrue(tl.done)
        self.assertGreaterEqual(tl.duration, 0.05)
        self.assertLessEqual(tl.duration, elapsed + 0.1)

    def test_concurrent_install_timer_no_crosstalk(self):
        # End-to-end shape of the crash reported in
        # https://github.com/numba/numba/issues/10564 : many threads
        # installing "numba:compiler_lock" timers while acquiring the
        # global compiler lock. Each scoped timer must observe exactly its
        # own acquire/release pair: no crash, no contamination, and the
        # callback must fire exactly once per round with a positive float.
        from numba.core.compiler_lock import global_compiler_lock

        nthreads = 8
        nrounds = 200
        barrier = threading.Barrier(nthreads)
        errors = []
        durations = []  # list.append() is atomic in CPython

        def worker():
            try:
                barrier.wait()
                for _ in range(nrounds):
                    with ev.install_timer("numba:compiler_lock",
                                          durations.append):
                        with global_compiler_lock:
                            pass
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(nthreads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertFalse(errors)
        # Every installed timer observed its paired events exactly once.
        self.assertEqual(len(durations), nthreads * nrounds)
        for d in durations:
            self.assertIsInstance(d, float)
            self.assertGreater(d, 0)


if __name__ == "__main__":
    unittest.main()
