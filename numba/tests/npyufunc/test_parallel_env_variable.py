from __future__ import absolute_import, print_function, division
from numba import unittest_support as unittest
from numba.npyufunc.parallel import get_thread_count
from os import environ as env
from numba import config


class TestParallelEnvVariable(unittest.TestCase):
    """
    Tests environment variables related to the underlying "parallel"
    functions for npyufuncs.
    """

    _numba_parallel_test_ = False

    def test_num_threads_variable(self):
        """
        Tests the NUMBA_NUM_THREADS env variable behaves as expected.
        """
        key = 'NUMBA_NUM_THREADS'
        current = str(getattr(env, key, config.NUMBA_DEFAULT_NUM_THREADS))
        threads = "3154"
        env[key] = threads
        config.reload_config()
        try:
            self.assertEqual(threads, str(get_thread_count()))
            self.assertEqual(threads, str(config.NUMBA_NUM_THREADS))
        finally:
            # reset the env variable/set to default
            env[key] = current
            config.reload_config()

if __name__ == '__main__':
    unittest.main()
