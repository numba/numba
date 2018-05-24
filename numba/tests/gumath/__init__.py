from numba import unittest_support as unittest

from os.path import dirname
from unittest.suite import TestSuite

from numba.testing import load_testsuite

def load_tests(loader, tests, pattern):
    suite = TestSuite()
    try:
        import gumath
    except ImportError:
        pass
    else:
        suite.addTests(load_testsuite(loader, dirname(__file__)))
    return suite
