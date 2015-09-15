from __future__ import print_function, division, absolute_import
from numba.testing import discover_tests, run_tests
import sys


def test():
    suite = discover_tests("numba.hsa.tests.hsapy")
    return run_tests(suite).wasSuccessful()


if __name__ == '__main__':
    sys.exit(0 if test() else 1)
