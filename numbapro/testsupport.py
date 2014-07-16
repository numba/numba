from numba import unittest_support as unittest
from numba.testing import discover_tests, run_tests


def runtests(modname):
    suite = discover_tests(modname)
    return run_tests(suite).wasSuccessful()

