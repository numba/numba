from numba.testing import unittest
from numba.testing import load_testsuite
from numba import cuda
from os.path import dirname, join


def load_tests(loader, tests, pattern):
    suite = unittest.TestSuite()
    this_dir = dirname(__file__)
    suite.addTests(load_testsuite(loader, join(this_dir, 'nocuda')))
    if cuda.is_available():
        # Ensure that cudart.so is loaded and the list of supported compute
        # capabilities in the nvvm module is populated before a fork. This is
        # needed because some compilation tests don't require a CUDA context,
        # but do use NVVM to compile, and it is required that libcudart.so
        # should be loaded before a fork (note that the requirement is not
        # explicitly documented).
        cuda.cudadrv.nvvm.get_supported_ccs()
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
