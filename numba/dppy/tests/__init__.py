from numba.testing import SerialSuite
from numba.testing import load_testsuite
from os.path import dirname, join
import dppy.ocldrv as ocldrv

def load_tests(loader, tests, pattern):

    suite = SerialSuite()
    this_dir = dirname(__file__)

    if ocldrv.is_available():
        suite.addTests(load_testsuite(loader, join(this_dir, 'dppy')))
    else:
        print("skipped DPPY tests")

    return suite
