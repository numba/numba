from numba.testing import SerialSuite
from numba.testing import load_testsuite
from numba import cuda
from os.path import dirname, join


def load_tests(loader, tests, pattern):

    suite = SerialSuite()
    this_dir = dirname(__file__)
    suite.addTests(load_testsuite(loader, join(this_dir, 'nocuda')))
    if cuda.is_available():
        suite.addTests(load_testsuite(loader, join(this_dir, 'cudasim')))
        gpus = cuda.list_devices()
        if gpus and gpus[0].compute_capability >= (2, 0):
            suite.addTests(load_testsuite(loader, join(this_dir, 'cudadrv')))
            suite.addTests(load_testsuite(loader, join(this_dir, 'cudapy')))
        else:
            print("skipped CUDA tests because GPU CC < 2.0")
    else:
        print("skipped CUDA tests")
    return suite
