from numba.testing import SerialSuite
from numba.testing import load_testsuite
from numba import roc
from os.path import dirname, join

def load_tests(loader, tests, pattern):

    suite = SerialSuite()
    this_dir = dirname(__file__)

    if roc.is_available():
        suite.addTests(load_testsuite(loader, join(this_dir, 'hsadrv')))
        suite.addTests(load_testsuite(loader, join(this_dir, 'hsapy')))

    else:
        print("skipped HSA tests")
    return suite
