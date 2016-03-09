import traceback
import multiprocessing
import numpy as np
from numba import cuda
from numba import unittest_support as unittest
from numba.cuda.testing import skip_on_cudasim

try:
    from concurrent.futures import ThreadPoolExecutor
except ImportError:
    has_concurrent_futures = False
else:
    has_concurrent_futures = True


has_mp_get_context = hasattr(multiprocessing, 'get_context')


def check_concurrent_compiling():
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


def spawn_process_entry(q):
    try:
        check_concurrent_compiling()
    except:
        msg = traceback.format_exc()
        q.put('\n'.join(['', '=' * 80, msg]))
    else:
        q.put(None)


@skip_on_cudasim('disabled for cudasim')
class TestMultiThreadCompiling(unittest.TestCase):

    @unittest.skipIf(not has_concurrent_futures, "no concurrent.futures")
    def test_concurrent_compiling(self):
        check_concurrent_compiling()

    @unittest.skipIf(not has_mp_get_context, "no multiprocessing.get_context")
    def test_spawn_concurrent_compilation(self):
        # force CUDA context init
        cuda.get_current_device()
        # use "spawn" to avoid inheriting the CUDA context
        ctx = multiprocessing.get_context('spawn')

        q = ctx.Queue()
        p = ctx.Process(target=spawn_process_entry, args=(q,))
        p.start()
        try:
            err = q.get()
        finally:
            p.join()
        if err is not None:
            raise AssertionError(err)
        self.assertEqual(p.exitcode, 0, 'test failed in child process')


if __name__ == '__main__':
    unittest.main()
