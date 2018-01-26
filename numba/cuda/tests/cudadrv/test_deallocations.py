from __future__ import division

from contextlib import contextmanager

import numpy as np

from numba import cuda, config
from numba.cuda.testing import unittest, skip_on_cudasim, SerialMixin
from numba.tests.support import captured_stderr


@skip_on_cudasim('not supported on CUDASIM')
class TestDeallocation(SerialMixin, unittest.TestCase):
    def test_max_pending_count(self):
        # get deallocation manager and flush it
        deallocs = cuda.current_context().deallocations
        deallocs.clear()
        self.assertEqual(len(deallocs), 0)
        # deallocate to maximum count
        for i in range(config.CUDA_DEALLOCS_COUNT):
            cuda.to_device(np.arange(1))
            self.assertEqual(len(deallocs), i + 1)
        # one more to trigger .clear()
        cuda.to_device(np.arange(1))
        self.assertEqual(len(deallocs), 0)

    def test_max_pending_bytes(self):
        # get deallocation manager and flush it
        ctx = cuda.current_context()
        deallocs = ctx.deallocations
        deallocs.clear()
        self.assertEqual(len(deallocs), 0)

        mi = ctx.get_memory_info()

        max_pending = 10**6  # 1MB
        old_ratio = config.CUDA_DEALLOCS_RATIO
        try:
            # change to a smaller ratio
            config.CUDA_DEALLOCS_RATIO = max_pending / mi.total
            self.assertEqual(deallocs._max_pending_bytes, max_pending)

            # deallocate half the max size
            cuda.to_device(np.ones(max_pending // 2, dtype=np.int8))
            self.assertEqual(len(deallocs), 1)

            # deallocate another remaining
            cuda.to_device(np.ones(max_pending - deallocs._size, dtype=np.int8))
            self.assertEqual(len(deallocs), 2)

            # another byte to trigger .clear()
            cuda.to_device(np.ones(1, dtype=np.int8))
            self.assertEqual(len(deallocs), 0)
        finally:
            # restore old ratio
            config.CUDA_DEALLOCS_RATIO = old_ratio


@skip_on_cudasim("defer_cleanup has no effect in CUDASIM")
class TestDeferCleanup(SerialMixin, unittest.TestCase):
    def test_basic(self):
        harr = np.arange(5)
        darr1 = cuda.to_device(harr)
        deallocs = cuda.current_context().deallocations
        deallocs.clear()
        self.assertEqual(len(deallocs), 0)
        with cuda.defer_cleanup():
            darr2 = cuda.to_device(harr)
            del darr1
            self.assertEqual(len(deallocs), 1)
            del darr2
            self.assertEqual(len(deallocs), 2)
            deallocs.clear()
            self.assertEqual(len(deallocs), 2)

        deallocs.clear()
        self.assertEqual(len(deallocs), 0)

    def test_nested(self):
        harr = np.arange(5)
        darr1 = cuda.to_device(harr)
        deallocs = cuda.current_context().deallocations
        deallocs.clear()
        self.assertEqual(len(deallocs), 0)
        with cuda.defer_cleanup():
            with cuda.defer_cleanup():
                darr2 = cuda.to_device(harr)
                del darr1
                self.assertEqual(len(deallocs), 1)
                del darr2
                self.assertEqual(len(deallocs), 2)
                deallocs.clear()
                self.assertEqual(len(deallocs), 2)
            deallocs.clear()
            self.assertEqual(len(deallocs), 2)

        deallocs.clear()
        self.assertEqual(len(deallocs), 0)

    def test_exception(self):
        harr = np.arange(5)
        darr1 = cuda.to_device(harr)
        deallocs = cuda.current_context().deallocations
        deallocs.clear()
        self.assertEqual(len(deallocs), 0)

        class CustomError(Exception):
            pass

        with self.assertRaises(CustomError):
            with cuda.defer_cleanup():
                darr2 = cuda.to_device(harr)
                del darr2
                self.assertEqual(len(deallocs), 1)
                deallocs.clear()
                self.assertEqual(len(deallocs), 1)
                raise CustomError
        deallocs.clear()
        self.assertEqual(len(deallocs), 0)
        del darr1
        self.assertEqual(len(deallocs), 1)
        deallocs.clear()
        self.assertEqual(len(deallocs), 0)


class TestDeferCleanupAvail(SerialMixin, unittest.TestCase):
    def test_context_manager(self):
        # just make sure the API is available
        with cuda.defer_cleanup():
            pass


@skip_on_cudasim('not supported on CUDASIM')
class TestDel(SerialMixin, unittest.TestCase):
    """
    Ensure resources are deleted properly without ignored exception.
    """
    @contextmanager
    def check_ignored_exception(self, ctx):
        with captured_stderr() as cap:
            yield
            ctx.deallocations.clear()
        self.assertFalse(cap.getvalue())

    def test_stream(self):
        ctx = cuda.current_context()
        stream = ctx.create_stream()
        with self.check_ignored_exception(ctx):
            del stream

    def test_event(self):
        ctx = cuda.current_context()
        event = ctx.create_event()
        with self.check_ignored_exception(ctx):
            del event

    def test_pinned_memory(self):
        ctx = cuda.current_context()
        mem = ctx.memhostalloc(32)
        with self.check_ignored_exception(ctx):
            del mem

    def test_mapped_memory(self):
        ctx = cuda.current_context()
        mem = ctx.memhostalloc(32, mapped=True)
        with self.check_ignored_exception(ctx):
            del mem

    def test_device_memory(self):
        ctx = cuda.current_context()
        mem = ctx.memalloc(32)
        with self.check_ignored_exception(ctx):
            del mem


if __name__ == '__main__':
    unittest.main()