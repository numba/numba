from numba.testing import SerialSuite
from numba.testing import load_testsuite
from os.path import dirname, join


import numba.dppl_config as dppl_config

def load_tests(loader, tests, pattern):

    suite = SerialSuite()
    this_dir = dirname(__file__)

    if dppl_config.dppl_present:
        suite.addTests(load_testsuite(loader, join(this_dir, 'dppl')))
    else:
        print("skipped DPPL tests")

    return suite
