"""
This subpackage contains CUDA tests that can be executed without a CUDA driver.
"""

from numba.tests import SerialSuite

def load_tests(loader, tests, pattern):
    suite = loader.discover("numba.cuda.tests.nocuda")
    return SerialSuite(suite)
