import os
import multiprocessing as mp

import numpy as np

from numba import ocl
from numba import unittest_support as unittest
from numba.ocl.testing import skip_on_oclsim

has_mp_get_context = hasattr(mp, 'get_context')
is_unix = os.name == 'posix'


def fork_test(q):
    from numba.ocl.ocldrv.error import OclDriverError
    try:
        ocl.to_device(np.arange(1))
    except OclDriverError as e:
        q.put(e)
    else:
        q.put(None)


class TestMultiprocessing(unittest.TestCase):
    @unittest.skipUnless(has_mp_get_context, 'requires mp.get_context')
    @unittest.skipUnless(is_unix, 'requires Unix')
    def test_fork(self):
        """
        Test fork detection.
        """
        ocl.current_context()  # force ocl initialize
        # fork in process that also uses OCL
        ctx = mp.get_context('fork')
        q = ctx.Queue()
        proc = ctx.Process(target=fork_test, args=[q])
        proc.start()
        exc = q.get()
        proc.join()
        # there should be an exception raised in the child process
        self.assertIsNotNone(exc)
        self.assertIn('OCL initialized before forking', str(exc))


if __name__ == '__main__':
    unittest.main()
