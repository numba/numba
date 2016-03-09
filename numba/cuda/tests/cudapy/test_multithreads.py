from numba import cuda
import numpy as np
from numba import unittest_support as unittest
from numba.cuda.testing import skip_on_cudasim

try:
    from concurrent.futures import ThreadPoolExecutor
except ImportError:
    has_thread_pool = False
else:
    has_thread_pool = True


@skip_on_cudasim('disabled for cudasim')
class TestMultiGPUContext(unittest.TestCase):

    @unittest.skipIf(not has_thread_pool, "no concurrent.futures")
    def test_concurrent_compiling(self):
        @cuda.jit
        def foo(x):
            x[0] += 1

        def use_foo(x):
            foo(x)
            return x

        arrays = [np.arange(10) for i in range(10)]
        expected = np.arange(10)
        expected[0] += 1
        with ThreadPoolExecutor(max_workers=4) as e:
            for ary in e.map(use_foo, arrays):
                np.testing.assert_equal(ary, expected)


if __name__ == '__main__':
    unittest.main()
