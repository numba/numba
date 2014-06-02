from numba import unittest_support as unittest
from numba.testing import discover_tests, run_tests


def runtests(modname, kwargs):
    suite = discover_tests(modname)
    return run_tests(suite, **kwargs).wasSuccessful()

