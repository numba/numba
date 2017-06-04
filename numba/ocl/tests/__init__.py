from numba.testing import SerialSuite
from numba.testing import load_testsuite
from numba import ocl
from os.path import dirname, join


def load_tests(loader, tests, pattern):

    suite = SerialSuite()
    this_dir = dirname(__file__)
    suite.addTests(load_testsuite(loader, join(this_dir, 'noocl')))
    if ocl.is_available():
        gpus = ocl.list_devices()
        if gpus and gpus[0].compute_capability >= (2, 0):
            suite.addTests(load_testsuite(loader, join(this_dir, 'ocldrv')))
            suite.addTests(load_testsuite(loader, join(this_dir, 'oclpy')))
        else:
            print("skipped OpenCL tests because driver version < 2.0")
    else:
        print("skipped OpenCL tests")
    return suite
