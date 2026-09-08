import numpy as np

from numba import cuda, vectorize, guvectorize
from numba.np.numpy_support import from_dtype
from numba.cuda.testing import CUDATestCase, skip_on_cudasim
import unittest


# Datetime/timedelta units exercised by the tests below. This covers both
# calendar-based units (Y, M) and units with a fixed conversion factor (D
# and finer) - see issue #5530, which reported that only datetime64[D] was
# tested on CUDA.
DATETIME_UNITS = ('Y', 'M', 'D', 'h', 'm', 's', 'ms', 'us', 'ns')


class TestCudaDateTime(CUDATestCase):
    def _datetime_array(self, unit):
        arr = np.arange('2005-02', '2006-02', dtype='datetime64[D]')
        return arr.astype('datetime64[%s]' % unit)

    def test_basic_datetime_kernel(self):
        @cuda.jit
        def foo(start, end, delta):
            for i in range(cuda.grid(1), delta.size, cuda.gridsize(1)):
                delta[i] = end[i] - start[i]

        for unit in DATETIME_UNITS:
            with self.subTest(unit=unit):
                arr1 = self._datetime_array(unit)
                arr2 = arr1 + np.random.randint(0, 10000, arr1.size)
                delta = np.zeros_like(arr1, dtype='timedelta64[%s]' % unit)

                foo[1, 32](arr1, arr2, delta)

                self.assertPreciseEqual(delta, arr2 - arr1)

    def test_scalar_datetime_kernel(self):
        @cuda.jit
        def foo(dates, target, delta, matches, outdelta):
            for i in range(cuda.grid(1), matches.size, cuda.gridsize(1)):
                matches[i] = dates[i] == target
                outdelta[i] = dates[i] - delta

        for unit in DATETIME_UNITS:
            with self.subTest(unit=unit):
                arr1 = self._datetime_array(unit)
                target = arr1[5]           # datetime
                delta = arr1[6] - arr1[5]  # timedelta
                matches = np.zeros_like(arr1, dtype=np.bool_)
                outdelta = np.zeros_like(arr1, dtype='datetime64[%s]' % unit)

                foo[1, 32](arr1, target, delta, matches, outdelta)

                # Coarser units (e.g. Y, M) can map multiple entries of
                # arr1 onto the same value as target, so compare against
                # the full expected match array rather than assuming a
                # single match at index 5.
                self.assertPreciseEqual(matches, arr1 == target)
                self.assertPreciseEqual(outdelta, arr1 - delta)

    @skip_on_cudasim('ufunc API unsupported in the simulator')
    def test_ufunc(self):
        for unit in DATETIME_UNITS:
            with self.subTest(unit=unit):
                datetime_t = from_dtype(np.dtype('datetime64[%s]' % unit))

                @vectorize([(datetime_t, datetime_t)], target='cuda')
                def timediff(start, end):
                    return end - start

                arr1 = self._datetime_array(unit)
                arr2 = arr1 + np.random.randint(0, 10000, arr1.size)

                delta = timediff(arr1, arr2)

                self.assertPreciseEqual(delta, arr2 - arr1)

    @skip_on_cudasim('ufunc API unsupported in the simulator')
    def test_gufunc(self):
        for unit in DATETIME_UNITS:
            with self.subTest(unit=unit):
                datetime_t = from_dtype(np.dtype('datetime64[%s]' % unit))
                timedelta_t = from_dtype(np.dtype('timedelta64[%s]' % unit))

                @guvectorize(
                    [(datetime_t, datetime_t, timedelta_t[:])],
                    '(),()->()', target='cuda')
                def timediff(start, end, out):
                    out[0] = end - start

                arr1 = self._datetime_array(unit)
                arr2 = arr1 + np.random.randint(0, 10000, arr1.size)

                delta = timediff(arr1, arr2)

                self.assertPreciseEqual(delta, arr2 - arr1)

    @skip_on_cudasim('no .copy_to_host() in the simulator')
    def test_datetime_view_as_int64(self):
        for unit in DATETIME_UNITS:
            with self.subTest(unit=unit):
                arr = self._datetime_array(unit)
                darr = cuda.to_device(arr)
                viewed = darr.view(np.int64)
                self.assertPreciseEqual(
                    arr.view(np.int64), viewed.copy_to_host())
                self.assertEqual(viewed.gpu_data, darr.gpu_data)

    @skip_on_cudasim('no .copy_to_host() in the simulator')
    def test_timedelta_view_as_int64(self):
        for unit in DATETIME_UNITS:
            with self.subTest(unit=unit):
                arr = self._datetime_array(unit)
                arr = arr - (arr - 1)
                self.assertEqual(
                    arr.dtype, np.dtype('timedelta64[%s]' % unit))
                darr = cuda.to_device(arr)
                viewed = darr.view(np.int64)
                self.assertPreciseEqual(
                    arr.view(np.int64), viewed.copy_to_host())
                self.assertEqual(viewed.gpu_data, darr.gpu_data)


if __name__ == '__main__':
    unittest.main()
