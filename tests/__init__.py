from numba.testing import SerialSuite
from numba.testing import load_testsuite
from os.path import dirname, join


import numba.dppy_config as dppy_config

def load_tests(loader, tests, pattern):

    suite = SerialSuite()
    this_dir = dirname(__file__)

    if dppy_config.dppy_present and dppy_config.is_available():
        suite.addTests(load_testsuite(loader, join(this_dir, 'dppy')))
    else:
        print("skipped DPPY tests")

    return suite
