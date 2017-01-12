from numba.testing import SerialSuite
from numba.testing import load_testsuite
import os

def load_tests(loader, tests, pattern):
    return SerialSuite(load_testsuite(loader, os.path.dirname(__file__)))
